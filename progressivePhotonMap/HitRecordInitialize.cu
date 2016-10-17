#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include "ppm.h"

using namespace optix;

rtBuffer<HitRecord, 3>           rtpass_output_buffer;
rtDeclareVariable(float,         rtpass_default_radius2, , );

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );

RT_PROGRAM void kernel()
{
    HitRecord rec;
    rec.normal=rec.position=make_float3(0.0f);

    rec.flags = PPM_NULL;
    rec.radius2 = rtpass_default_radius2;
    rec.photon_count = 0;
    rec.accum_atten = 0.0f;
    rec.flux = make_float3(0.0f, 0.0f, 0.0f);
    rec.accum_atten = 0.0f;
    rec.volumetricRadiance = make_float3(0.0f);
    rtpass_output_buffer[make_uint3(launchIndex.x, launchIndex.y,0)] = rec;
}