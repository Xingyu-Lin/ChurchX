/*
* Copyright (c) 2013 Opposite Renderer
* For the full copyright and license information, please view the LICENSE.txt
* file that was distributed with this source code.
*/

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "random.h"
#include "ppm.h"

using namespace optix;

//
// Scene wide variables
//
rtDeclareVariable(rtObject, volumetricPhotonsRoot, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );

rtDeclareVariable(float, sigma_a, , );
rtDeclareVariable(float, sigma_s, , );

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PhotonPRD, photonPrd, rtPayload, );
rtDeclareVariable(HitPRD, hitPrd, rtPayload, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );
rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, );
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, );

rtBuffer<Photon, 1> volumetricPhotons;

RT_PROGRAM void closestHitRadiance()
{
    const float sigma_t = sigma_a + sigma_s;
    float3 worldShadingNormal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shadingNormal));
    float3 hitPoint = ray.origin + tHit*ray.direction;
    bool isHitFromOutside = hitFromOutside(ray.direction, worldShadingNormal);
    double tHitStack = tHit + 0.1 - 0.1; // Important, prevents compiler optimization on variable

    /*OPTIX_DEBUG_PRINT(0, "Hit media (%.2f %.2f %.2f) %s (attn: %.2f %.2f  %.2f)\n", hitPoint.x, hitPoint.y, hitPoint.z, isHitFromOutside ? "outside" : "inside",
        radiancePrd.attenuation.x, radiancePrd.attenuation.y, radiancePrd.attenuation.z);*/

    if(isHitFromOutside)
    {
        float3 attenSaved = hitPrd.attenuation + 0.1 - 0.1; // Important, prevents compiler optimization on variable

        // Send ray through the medium
        Ray newRay(hitPoint, ray.direction, radiance_in_participating_medium, 0.01);
        rtTrace(top_object, newRay, hitPrd);

        float distance = hitPrd.lastTHit;
        float transmittance = exp(-distance*sigma_t);

        VolumetricRadiancePRD volRadiancePrd;
        volRadiancePrd.radiance = make_float3(0);
        volRadiancePrd.numHits = 0;
        volRadiancePrd.sigma_t = sigma_t;
        volRadiancePrd.sigma_s = sigma_s;

        // Get volumetric radiance

        Ray ray(hitPoint, ray.direction, volumetric_radiance, 0.0000001, distance);
        rtTrace(volumetricPhotonsRoot, ray, volRadiancePrd);

        // Multiply existing volumetric transmittance with current transmittance, and add gathered volumetric radiance
        // from this path
        float3 tmp = hitPrd.volumetricRadiance;
        hitPrd.volumetricRadiance *= transmittance;
        hitPrd.volumetricRadiance += attenSaved*volRadiancePrd.radiance;
        hitPrd.attenuation *= transmittance;

    }
    else
    {
        // We are escaping the boundary of the participating medium, so we'll compute the attenuation and volumetric radiance for the remaining path
        // and deliver it to a parent stack frame.
        Ray newRay = Ray(hitPoint, ray.direction, rtpass_ray_type, 0.01);
        //if (hitPrd.volumetricRadiance.x>0)
          //  rtPrintf("hello\n");
        rtTrace(top_object, newRay, hitPrd);
    }
    hitPrd.lastTHit = tHitStack;

}

static __device__ __inline__ float rnd_float_from_uint2(uint2& prev)
{
    return rnd(prev.x) / 2 + rnd(prev.y) / 2;
}

static __device__ __inline__ float2 rnd_from_uint2(uint2& prev)
{
    return make_float2(rnd(prev.x), rnd(prev.y));
}

/*
//
*/

RT_PROGRAM void closestHitPhoton()
{
    const float sigma_t = sigma_a + sigma_s;

    photonPrd.ray_depth++;

    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    float3 hitPoint = ray.origin + tHit*ray.direction;
    bool hitInside = (dot(worldShadingNormal, ray.direction) > 0);

    // If we hit from the inside with a PHOTON_IN_PARTICIPATING_MEDIUM ray, we have escaped the boundry of the medium.
    // We move the ray just a tad to the outside and continue ray tracing there
    if(hitInside && ray.ray_type == photon_in_participating_medium)
    {
        //OPTIX_DEBUG_PRINT(photonPrd.depth-1, "Hit medium P(%.2f %.2f %.2f) from inside: move past\n", hitPoint.x, hitPoint.y, hitPoint.z);
        Ray newRay = Ray(hitPoint+0.0001*ray.direction, ray.direction, ppass_and_gather_ray_type, 0.001, RT_DEFAULT_MAX);
        rtTrace(top_object, newRay, photonPrd);
        return;
    }

    float sample = rnd_float_from_uint2(photonPrd.sample);

    float scatterLocationT = - logf(1-sample)/sigma_t;
    float3 scatterPosition = hitPoint + scatterLocationT*ray.direction;
    int depth = photonPrd.ray_depth;

    // We need to see if anything obstructs the ray in the interval from the hitpoint to the scatter location.
    // If nothings obstructs then we scatter at eventPosition. Otherwise, the photon continues on its path and we don't do anything
    // when we return to this stack frame. We keep the photonPRD depth on the stack to compare it when the rtTrace returns.

    Ray newRay(hitPoint, ray.direction, photon_in_participating_medium, 0.001, scatterLocationT);
    rtTrace(top_object, newRay, photonPrd);

    // If depth is unmodified, no surface was hit from hitpoint to scatterLocation, so we store it as a scatter event.
    // We also scatter a photon in a new direction sampled by the phase function at this location.

    if(depth == photonPrd.ray_depth)
    {
        const float scatterAlbedo = sigma_s/sigma_t;

        if (sample >= scatterAlbedo)
        {
            return;
        }
        //photonPrd.power *= scatterAlbedo;

        // Store photon at scatter location

        //if(photonPrd.numStoredPhotons < maxPhotonDepositsPerEmitted)
        {
            int volumetricPhotonIdx = photonPrd.pm_index % 200000; //TODO:
            volumetricPhotons[volumetricPhotonIdx].power = photonPrd.energy;
            volumetricPhotons[volumetricPhotonIdx].position = scatterPosition;
            atomicAdd(&volumetricPhotons[volumetricPhotonIdx].numDeposits, 1);
        }

        // Check if we have gone above max number of photons or stack depth
        if(photonPrd.ray_depth >=  15) //TODO:
        {
            return;
        }

        // Create the scattered ray with a direction given by importance sampling of the phase function

        float3 scatterDirection = sampleUnitSphere(rnd_from_uint2(photonPrd.sample));

        //OPTIX_DEBUG_PRINT(photonPrd.depth-1, "Not interrupted. Store, scatter P(%.2f %.2f %.2f) D(%.2f %.2f %.2f)\n", scatterPosition.x, scatterPosition.y, scatterPosition.z,
          //                scatterDirection.x, scatterDirection.y, scatterDirection.z);

        Ray scatteredRay(scatterPosition, scatterDirection, ppass_and_gather_ray_type, 0.001, RT_DEFAULT_MAX);
        rtTrace(top_object, scatteredRay, photonPrd);

    }
    else
    {
        //OPTIX_DEBUG_PRINT(depth-1, "Found surface in [0,t], no scatter!\n");
    }
}
