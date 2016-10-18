
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <optixu/optixu_math_namespace.h>

#define NUM_VOLUMETRIC_PHOTONS 20000
#define TOTAL_DISTANCE 1000
#define  FRAME 20 //divisable by 4
#define  FRAME_PACKED 15
#define  PPM_X         ( 1 << 0 )
#define  PPM_Y         ( 1 << 1 )
#define  PPM_Z         ( 1 << 2 )
#define  PPM_LEAF      ( 1 << 3 )
#define  PPM_NULL      ( 1 << 4 )

#define  PPM_IN_SHADOW ( 1 << 5 )
#define  PPM_OVERFLOW  ( 1 << 6 )
#define  PPM_HIT       ( 1 << 7 )


enum RayTypes
{
    rtpass_ray_type,
    ppass_and_gather_ray_type,
    shadow_ray_type,
    radiance_in_participating_medium,
    photon_in_participating_medium,
    volumetric_radiance,
    num_ray_type
};

struct PPMLight
{
  optix::uint   is_area_light;
  optix::float3 power;

  // For spotlight
  optix::float3 position;
  optix::float3 direction;
  float         radius;

  // Parallelogram
  optix::float3 anchor;
  optix::float3 v1;
  optix::float3 v2;
};

struct HitRecord
{
 // float3 ray_dir;          // rgp

  optix::float3 position;         //
  optix::float3 normal;           // Material shader
  optix::float3 attenuated_Kd;
  optix::uint   flags;

  float         radius2;          //
  float         photon_count;     // Client TODO: should be moved clientside?
  optix::float3 flux;             //
  float         accum_atten;
  optix::float3 volumetricRadiance[FRAME];
};


struct PackedHitRecord
{
  optix::float4 a;   // position.x, position.y, position.z, normal.x
  optix::float4 b;   // normal.y,   normal.z,   atten_Kd.x, atten_Kd.y
  optix::float4 c;   // atten_Kd.z, flags,      radius2,    photon_count
  optix::float4 d;   // flux.x,     flux.y,     flux.z,     accum_atten
  optix::float4 e[FRAME_PACKED];   // volumetricRadiance[i]
};


struct HitPRD
{
    optix::float3 attenuation;
    optix::uint   ray_depth;
    optix::float3 volumetricRadiance[FRAME];
    float lastTHit;
};


struct PhotonRecord
{
  optix::float3 position;
  optix::float3 normal;      // Pack this into 4 bytes
  optix::float3 ray_dir;
  optix::float3 energy;
  optix::uint   axis;
  float dist;
  optix::float2 pad;
};


struct PackedPhotonRecord
{
  optix::float4 a;   // position.x, position.y, position.z, normal.x
  optix::float4 b;   // normal.y,   normal.z,   ray_dir.x,  ray_dir.y
  optix::float4 c;   // ray_dir.z,  energy.x,   energy.y,   energy.z
  optix::float4 d;   // axis,       dist,    padding,    padding
};


struct PhotonPRD
{
  optix::float3 energy;
  optix::uint2  sample;
  optix::uint   pm_index;
  optix::uint   num_deposits;
  optix::uint   ray_depth;
  float dist; // Distance from the photon to the light source
};


struct ShadowPRD
{
  float attenuation;
};

struct VolumetricRadiancePRD
{
    float sigma_t;
    float sigma_s;
    optix::float3 radiance[FRAME];
    unsigned int numHits;
};

struct Photon
{
#ifdef __CUDACC__
    __device__ __inline Photon(const optix::float3 & power, const optix::float3 & position, const optix::float3 & rayDirection, const optix::uint & objectId)
		: power(power), position(position), rayDirection(rayDirection), objectId(objectId)
	{

	}

	__device__ __inline Photon(void)
	{

	}
#endif
    optix::float3 power;
    optix::float3 position;
    optix::float3 rayDirection;
    optix::uint   objectId;
    optix::uint numDeposits;
    float dist;
};