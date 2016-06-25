#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include "ppm.h"



using namespace optix;

rtBuffer<Photon, 1> volumetricPhotons;
rtDeclareVariable(uint1, launchIndex, rtLaunchIndex, );

RT_PROGRAM void kernel()
{
    Photon photon = Photon(make_float3(0), make_float3(0), make_float3(0), 1);
    photon.numDeposits = 0;
    volumetricPhotons[launchIndex.x] = photon;
}

