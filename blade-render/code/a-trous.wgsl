#include "camera.inc.wgsl"
#include "gbuf.inc.wgsl"
#include "quaternion.inc.wgsl"
#include "surface.inc.wgsl"

// Spatio-temporal variance-guided filtering
// https://research.nvidia.com/sites/default/files/pubs/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A//svgf_preprint.pdf

// RGB is illumination; alpha is luminance variance. Unfiltered temporal RGB
// and variance are retained in the current-frame target as next frame's history.

struct Params {
    extent: vec2<i32>,
    temporal_weight: f32,
    iteration: u32,
    use_motion_vectors: u32,
    filter_specular: u32,
}

var<uniform> camera: CameraParams;
var<uniform> prev_camera: CameraParams;
var<uniform> params: Params;
var t_depth: texture_2d<f32>;
var t_prev_depth: texture_2d<f32>;
var t_flat_normal: texture_2d<f32>;
var t_prev_flat_normal: texture_2d<f32>;
var t_basis: texture_2d<f32>;
var t_prev_basis: texture_2d<f32>;
var t_specular_f0: texture_2d<f32>;
var t_prev_specular_f0: texture_2d<f32>;
var t_motion: texture_2d<f32>;
var input: texture_2d<f32>;
var output: texture_storage_2d<rgba16float, read_write>;

const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);
const MIN_WEIGHT: f32 = 0.01;

fn read_surface(pixel: vec2<i32>) -> Surface {
    var surface = Surface();
    surface.depth = textureLoad(t_depth, pixel, 0).x;
    if (surface.depth > 0.0) {
        surface.flat_normal = normalize(textureLoad(t_flat_normal, pixel, 0).xyz);
        surface.basis = normalize(textureLoad(t_basis, pixel, 0));
        let specular = textureLoad(t_specular_f0, pixel, 0);
        surface.roughness = specular.w;
        surface.specular_f0 = specular.xyz;
    }
    return surface;
}
fn read_prev_surface(pixel: vec2<i32>) -> Surface {
    var surface = Surface();
    surface.depth = textureLoad(t_prev_depth, pixel, 0).x;
    if (surface.depth > 0.0) {
        surface.flat_normal = normalize(textureLoad(t_prev_flat_normal, pixel, 0).xyz);
        surface.basis = normalize(textureLoad(t_prev_basis, pixel, 0));
        let specular = textureLoad(t_prev_specular_f0, pixel, 0);
        surface.roughness = specular.w;
        surface.specular_f0 = specular.xyz;
    }
    return surface;
}

fn get_prev_pixel(pixel: vec2<i32>, pos_world: vec3<f32>) -> vec2<f32> {
    if (USE_MOTION_VECTORS && params.use_motion_vectors != 0u) {
        let motion = textureLoad(t_motion, pixel, 0).xy / MOTION_SCALE;
        return vec2<f32>(pixel) + 0.5 + motion;
    } else {
        return get_projected_pixel_float(prev_camera, pos_world);
    }
}

fn normal_weight(a: Surface, b: Surface) -> f32 {
    let na = qrot(a.basis, vec3<f32>(0.0, 0.0, 1.0));
    let nb = qrot(b.basis, vec3<f32>(0.0, 0.0, 1.0));
    var exponent = 4.0;
    var material_weight = 1.0;
    if (params.filter_specular != 0u) {
        let roughness = min(a.roughness, b.roughness);
        exponent = mix(256.0, 4.0, roughness * roughness);
        material_weight = exp(-8.0 * abs(a.roughness - b.roughness));
        material_weight *= exp(-8.0 * length(a.specular_f0 - b.specular_f0));
    }
    return compare_flat_normals(a.flat_normal, b.flat_normal)
        * pow(max(0.0, dot(na, nb)), exponent) * material_weight;
}

@compute @workgroup_size(8, 8)
fn temporal_accum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = vec2<i32>(global_id.xy);
    if (any(pixel >= params.extent)) {
        return;
    }

    let surface = read_surface(pixel);
    let cur_illumination = textureLoad(output, pixel).xyz;
    if (surface.depth == 0.0 || params.temporal_weight >= 1.0) {
        textureStore(output, pixel, vec4<f32>(cur_illumination, 0.0));
        return;
    }
    let pos_world = camera.position + surface.depth * get_ray_direction(camera, pixel);
    // considering all samples in 2x2 quad, to help with edges
    let center_pixel = get_prev_pixel(pixel, pos_world);
    let base = vec2<i32>(floor(center_pixel - vec2<f32>(0.5)));
    var prev_pixels = array<vec2<i32>, 4>(
        base, base + vec2<i32>(1, 0), base + vec2<i32>(1, 1), base + vec2<i32>(0, 1),
    );
    //Note: careful about the pixel center when there is a perfect match
    let w_bot_right = fract(center_pixel + vec2<f32>(0.5));
    var prev_weights = vec4<f32>(
        (1.0 - w_bot_right.x) * (1.0 - w_bot_right.y),
        w_bot_right.x * (1.0 - w_bot_right.y),
        w_bot_right.x * w_bot_right.y,
        (1.0 - w_bot_right.x) * w_bot_right.y,
    );

    var sum_weight = 0.0;
    var sum_ilm = vec4<f32>(0.0);
    var projected_distance = length(pos_world - prev_camera.position);
    if (USE_MOTION_VECTORS && params.use_motion_vectors != 0u) {
        projected_distance = textureLoad(t_motion, pixel, 0).z;
    }
    if (params.temporal_weight != 1.0) {
        //TODO: optimize depth load with a gather operation
        for (var i = 0; i < 4; i += 1) {
            let prev_pixel = prev_pixels[i];
            if (all(prev_pixel >= vec2<i32>(0)) && all(prev_pixel < params.extent)) {
                let prev_surface = read_prev_surface(prev_pixel);
                if (prev_surface.depth == 0.0 || normal_weight(surface, prev_surface) < 0.5) {
                    continue;
                }
                if (abs(prev_surface.depth - projected_distance) > max(0.001, 0.01 * projected_distance)) {
                    continue;
                }
                let w = prev_weights[i];
                sum_weight += w;
                let history = textureLoad(input, prev_pixel, 0);
                let luminance = dot(history.xyz, LUMA);
                sum_ilm += w * vec4<f32>(history.xyz, max(history.w, 0.0) + luminance * luminance);
            }
        }
    }

    let cur_luminocity = dot(cur_illumination, LUMA);
    var mixed_ilm = vec4<f32>(cur_illumination, cur_luminocity * cur_luminocity);
    if (sum_weight > MIN_WEIGHT) {
        let prev_ilm = sum_ilm / sum_weight;
        mixed_ilm = mix(mixed_ilm, prev_ilm, min(sum_weight, 1.0) * (1.0 - params.temporal_weight));
    }
    let mean_luminance = dot(mixed_ilm.xyz, LUMA);
    mixed_ilm.w = max(0.0, mixed_ilm.w - mean_luminance * mean_luminance);
    //Note: could also use HW blending for this
    textureStore(output, pixel, mixed_ilm);
}

const GAUSSIAN_WEIGHTS = vec2<f32>(0.44198, 0.27901);
const SIGMA_L: f32 = 4.0;
const EPSILON: f32 = 0.001;

fn compare_luminance(a_lum: f32, b_lum: f32, variance: f32) -> f32 {
    return exp(-abs(a_lum - b_lum) / (SIGMA_L * variance + EPSILON));
}

@compute @workgroup_size(8, 8)
fn atrous3x3(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let center = vec2<i32>(global_id.xy);
    if (any(center >= params.extent)) {
        return;
    }

    let center_ilm = textureLoad(input, center, 0);
    let center_luma = dot(center_ilm.xyz, LUMA);
    let center_suf = read_surface(center);
    if (center_suf.depth == 0.0) {
        textureStore(output, global_id.xy, center_ilm);
        return;
    }
    let center_position = camera.position + center_suf.depth * get_ray_direction(camera, center);
    var filtered_ilm = center_ilm;
    var neighbor_variance = 0.0;
    var neighbor_weight = 0.0;

    for (var yy=-1; yy<=1; yy+=1) {
        for (var xx=-1; xx<=1; xx+=1) {
            let p = center + vec2<i32>(xx, yy) * (1i << params.iteration);
            if (all(p == center) || any(p < vec2<i32>(0)) || any(p >= params.extent)) {
                continue;
            }

            //TODO: store in group-shared memory
            let surface = read_surface(p);
            if (surface.depth == 0.0) {
                continue;
            }
            var weight = GAUSSIAN_WEIGHTS[abs(xx)] * GAUSSIAN_WEIGHTS[abs(yy)];
            weight *= normal_weight(surface, center_suf);
            // Ray distance is not planar depth: even a flat wall's distance
            // changes across the image. Use a symmetric point-to-plane error
            // to preserve both sloped surfaces and nearby depth discontinuities.
            let position = camera.position + surface.depth * get_ray_direction(camera, p);
            let delta = position - center_position;
            let plane_error = max(abs(dot(delta, surface.flat_normal)), abs(dot(delta, center_suf.flat_normal)));
            // A smooth curved surface bends away from both tangent planes.
            // Account for that bend, but not for a gap between parallel faces.
            let curvature = abs(dot(delta, surface.flat_normal - center_suf.flat_normal));
            let depth_scale = max(0.0001, 0.001 * min(surface.depth, center_suf.depth) + curvature);
            weight *= exp(-plane_error / depth_scale);
            let other_ilm = textureLoad(input, p, 0);
            // The luminance gate must be symmetric. Using only the centre's
            // variance lets a noisy bright pixel accept a dark neighbour while
            // the dark pixel rejects the bright one, which systematically
            // moves radiance out of highlights and shadowed geometry.
            let variance = sqrt(max(center_ilm.w, other_ilm.w));
            weight *= compare_luminance(center_luma, dot(other_ilm.xyz, LUMA), variance);

            // Rejected neighbour weight stays on the centre instead of
            // renormalising the surviving neighbours. Every pair then applies
            // equal and opposite RGB deltas, so an A-trous pass conserves
            // linear radiance over the frame. The Gaussian neighbour weights
            // sum to less than one, keeping this a convex update.
            filtered_ilm = vec4<f32>(filtered_ilm.xyz + weight * (other_ilm.xyz - center_ilm.xyz), 0.0);
            neighbor_variance += weight * weight * max(other_ilm.w, 0.0);
            neighbor_weight += weight;
        }
    }

    filtered_ilm.w = (1.0 - neighbor_weight) * (1.0 - neighbor_weight) * max(center_ilm.w, 0.0) + neighbor_variance;
    textureStore(output, global_id.xy, filtered_ilm);
}
