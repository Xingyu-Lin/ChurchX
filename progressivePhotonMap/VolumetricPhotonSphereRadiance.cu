
#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include "ppm.h"

using namespace optix;

rtDeclareVariable(VolumetricRadiancePRD, volRadiancePrd, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );

rtDeclareVariable(float3, photonPosition, attribute photonPosition, ); 
rtDeclareVariable(float3, photonPower, attribute photonPower, ); 
rtDeclareVariable(uint, photonId, attribute photonId, ); 

rtDeclareVariable(float, volumetricRadius, ,); 

RT_PROGRAM void anyHitRadiance()
{
    float t = dot(photonPosition-ray.origin, ray.direction)/100;

    if(t < ray.tmax && t > ray.tmin)
    {
        volRadiancePrd.radiance += (1/(M_PIf*volumetricRadius*volumetricRadius)) * photonPower * exp(-volRadiancePrd.sigma_t*t) * (1.f/(4.f*M_PIf));
        //rtPrintf("%f %f %f %f\n", t, volumetricRadius, photonPower.x, (1/(M_PIf*volumetricRadius*volumetricRadius)) * photonPower.x * exp(-volRadiancePrd.sigma_t*t) * (1.f/(4.f*M_PIf)));
        volRadiancePrd.numHits++;
    }
    rtIgnoreIntersection();
}