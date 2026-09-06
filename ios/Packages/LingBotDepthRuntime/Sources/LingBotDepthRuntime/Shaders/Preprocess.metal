#include <metal_stdlib>
using namespace metal;

kernel void preprocessYCbCrDepth(
    texture2d<float, access::sample> yTexture [[texture(0)]],
    texture2d<float, access::sample> cbcrTexture [[texture(1)]],
    texture2d<float, access::sample> depthTexture [[texture(2)]],
    device float *image [[buffer(0)]],
    device float *depth [[buffer(1)]],
    device float *validity [[buffer(2)]],
    constant uint2 &outputSize [[buffer(3)]],
    constant float3x3 &modelToSource [[buffer(4)]],
    constant float2 &sourceSize [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= outputSize)) return;
    constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    constexpr sampler nearestSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    float3 source = modelToSource * float3(float2(gid) + 0.5, 1.0);
    float2 uv = source.xy / sourceSize;
    float y = yTexture.sample(linearSampler, uv).r;
    float2 chroma = cbcrTexture.sample(linearSampler, uv).rg - 0.5;
    float3 rgb = clamp(float3(y + 1.5748 * chroma.y,
                              y - 0.1873 * chroma.x - 0.4681 * chroma.y,
                              y + 1.8556 * chroma.x), 0.0, 1.0);
    uint index = gid.y * outputSize.x + gid.x;
    uint plane = outputSize.x * outputSize.y;
    image[index] = rgb.r;
    image[plane + index] = rgb.g;
    image[2 * plane + index] = rgb.b;
    float value = depthTexture.sample(nearestSampler, uv).r;
    bool valid = isfinite(value) && value > 0.01;
    depth[index] = valid ? value : 0.0;
    validity[index] = valid ? 1.0 : 0.0;
}
