//! Fallible, dedicated native images for clients which must reserve their exact
//! native storage before allocating it. The existing pooled resource API is unchanged.
use super::*;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeError {
    Invalid,
    Vulkan(vk::Result),
}
impl std::fmt::Display for NativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "native GPU: {self:?}")
    }
}
impl std::error::Error for NativeError {}
impl From<vk::Result> for NativeError {
    fn from(e: vk::Result) -> Self {
        Self::Vulkan(e)
    }
}
impl NativeError {
    pub fn is_device_lost(self) -> bool {
        self == Self::Vulkan(vk::Result::ERROR_DEVICE_LOST)
    }
}

pub struct NativeImagePlan {
    image: NativeImage,
    requirements: vk::MemoryRequirements,
}
pub struct NativeImage {
    context: Arc<Context>,
    raw: vk::Image,
    memory: vk::DeviceMemory,
    target_size: [u16; 2],
    format: crate::TextureFormat,
}
pub struct NativeImageView {
    image: Arc<NativeImage>,
    raw: vk::ImageView,
}
impl NativeImagePlan {
    /// Includes native alignment padding. Query this before allocating backing.
    pub fn reservation_bytes(&self) -> Result<u64, NativeError> {
        aligned_size(self.requirements.size, self.requirements.alignment)
    }
    pub fn allocate(mut self) -> Result<Arc<NativeImage>, NativeError> {
        let ctx = &self.image.context;
        let properties = unsafe {
            ctx.instance
                .core
                .get_physical_device_memory_properties(ctx.physical_device)
        };
        let index = (0..properties.memory_type_count)
            .find(|&i| {
                self.requirements.memory_type_bits & (1 << i) != 0
                    && properties.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or(NativeError::Invalid)?;
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(self.image.raw);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(self.requirements.size)
            .memory_type_index(index)
            .push_next(&mut dedicated);
        self.image.memory = unsafe { ctx.device.core.allocate_memory(&info, None)? };
        unsafe {
            ctx.device
                .core
                .bind_image_memory(self.image.raw, self.image.memory, 0)?;
        }
        Ok(Arc::new(self.image))
    }
}
fn aligned_size(size: u64, alignment: u64) -> Result<u64, NativeError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(NativeError::Invalid);
    }
    size.checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or(NativeError::Invalid)
}
impl NativeImage {
    /// Borrowed opaque handle; never pass it to `destroy_texture`.
    pub fn texture(&self) -> Texture {
        Texture {
            raw: self.raw,
            memory_handle: !0,
            target_size: self.target_size,
            format: self.format,
            external: None,
        }
    }
    pub fn create_view(self: &Arc<Self>) -> Result<NativeImageView, NativeError> {
        let info = vk::ImageViewCreateInfo::default()
            .image(self.raw)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(map_texture_format(self.format))
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let raw = unsafe { self.context.device.core.create_image_view(&info, None)? };
        Ok(NativeImageView {
            image: self.clone(),
            raw,
        })
    }
}
impl NativeImageView {
    /// Borrowed opaque handle; never pass it to `destroy_texture_view`.
    pub fn view(&self) -> TextureView {
        TextureView {
            raw: self.raw,
            target_size: self.image.target_size,
            aspects: crate::TexelAspects::COLOR,
        }
    }
    pub fn texture(&self) -> Texture {
        self.image.texture()
    }
}
impl Drop for NativeImageView {
    fn drop(&mut self) {
        unsafe {
            self.image
                .context
                .device
                .core
                .destroy_image_view(self.raw, None)
        };
    }
}
impl Drop for NativeImage {
    fn drop(&mut self) {
        unsafe {
            self.context.device.core.destroy_image(self.raw, None);
            if self.memory != vk::DeviceMemory::null() {
                self.context.device.core.free_memory(self.memory, None);
            }
        }
    }
}

/// A submission receipt remains available even when presentation fails after
/// queue submission. The caller must retire resources using this sync point.
pub struct NativeSubmission {
    pub sync_point: SyncPoint,
    pub presentation_error: Option<NativeError>,
}
impl Context {
    pub fn prepare_native_image(
        self: &Arc<Self>,
        desc: crate::TextureDesc,
    ) -> Result<NativeImagePlan, NativeError> {
        // Deliberately narrow additive API: owned 2D colour targets only.
        if desc.dimension != crate::TextureDimension::D2
            || desc.size.depth != 1
            || desc.size.width == 0
            || desc.size.height == 0
            || desc.size.width > u16::MAX as u32
            || desc.size.height > u16::MAX as u32
            || desc.array_layer_count != 1
            || desc.mip_level_count != 1
            || desc.external.is_some()
            || desc.format.aspects() != crate::TexelAspects::COLOR
            || !desc.sample_count.is_power_of_two()
            || self.sample_count_flags.as_raw() & desc.sample_count == 0
        {
            return Err(NativeError::Invalid);
        }
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(map_texture_format(desc.format))
            .extent(map_extent_3d(&desc.size))
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::from_raw(desc.sample_count))
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let raw = unsafe { self.device.core.create_image(&info, None)? };
        // Install RAII immediately, including all failure paths after the query.
        let image = NativeImage {
            context: self.clone(),
            raw,
            memory: vk::DeviceMemory::null(),
            target_size: [desc.size.width as u16, desc.size.height as u16],
            format: desc.format,
        };
        let requirements = unsafe { self.device.core.get_image_memory_requirements(raw) };
        Ok(NativeImagePlan {
            image,
            requirements,
        })
    }
    pub fn try_submit_native(
        &self,
        encoder: &mut CommandEncoder,
    ) -> Result<NativeSubmission, NativeError> {
        let raw_cmd_buf = encoder.try_finish_native()?;
        let mut queue = self.queue.lock().map_err(|_| NativeError::Invalid)?;
        let progress = queue
            .last_progress
            .checked_add(1)
            .ok_or(NativeError::Invalid)?;
        let command_buffers = [raw_cmd_buf];
        let wait_values = [0];
        let mut waits = [vk::Semaphore::null()];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let mut signals = [queue.timeline_semaphore, vk::Semaphore::null()];
        let values = [progress, 0];
        let (nw, ns) = if let Some(p) = &encoder.present {
            waits[0] = p.acquire_semaphore;
            signals[1] = p.present_semaphore;
            (1, 2)
        } else {
            (0, 1)
        };
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values[..nw])
            .signal_semaphore_values(&values[..ns]);
        let info = vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .wait_semaphores(&waits[..nw])
            .wait_dst_stage_mask(&stages[..nw])
            .signal_semaphores(&signals[..ns])
            .push_next(&mut timeline);
        unsafe {
            self.device
                .core
                .queue_submit(queue.raw, &[info], vk::Fence::null())?;
        }
        queue.last_progress = progress;
        let presentation_error = encoder.present.take().and_then(|p| {
            let swapchains = [p.swapchain];
            let indices = [p.image_index];
            let semaphores = [p.present_semaphore];
            let info = vk::PresentInfoKHR::default()
                .swapchains(&swapchains)
                .image_indices(&indices)
                .wait_semaphores(&semaphores);
            match self.device.swapchain.as_ref() {
                Some(s) => match unsafe { s.queue_present(queue.raw, &info) } {
                    Ok(_) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => None,
                    Err(e) => Some(e.into()),
                },
                None => Some(NativeError::Invalid),
            }
        });
        Ok(NativeSubmission {
            sync_point: SyncPoint { progress },
            presentation_error,
        })
    }
    /// A finite timeout is pending, never device loss or retirement.
    pub fn try_wait_native(&self, sp: &SyncPoint, timeout_ms: u32) -> Result<bool, NativeError> {
        let semaphore = self
            .queue
            .lock()
            .map_err(|_| NativeError::Invalid)?
            .timeline_semaphore;
        let semaphores = [semaphore];
        let values = [sp.progress];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        match unsafe {
            self.device
                .timeline_semaphore
                .wait_semaphores(&info, map_timeout(timeout_ms))
        } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_alignment_is_checked_before_allocation() {
        assert_eq!(aligned_size(4097, 4096), Ok(8192));
        assert_eq!(aligned_size(u64::MAX, 4096), Err(NativeError::Invalid));
        assert_eq!(aligned_size(4, 0), Err(NativeError::Invalid));
    }
    #[test]
    fn only_device_loss_proves_terminal_retirement() {
        assert!(!NativeError::Vulkan(vk::Result::TIMEOUT).is_device_lost());
        assert!(!NativeError::Vulkan(vk::Result::ERROR_OUT_OF_HOST_MEMORY).is_device_lost());
        assert!(NativeError::Vulkan(vk::Result::ERROR_DEVICE_LOST).is_device_lost());
    }
}
