#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "ppm.h"
#include "path_tracer.h"
#include "random.h"

using namespace optix;

//
// Scene wide variables
//
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );


//
// Ray generation program
//
rtBuffer<PhotonRecord, 1>        ppass_output_buffer;
rtBuffer<uint2, 2>               photon_rnd_seeds;
rtDeclareVariable(uint,          max_depth, , );
rtDeclareVariable(uint,          max_photon_count, , );
rtDeclareVariable(PPMLight,      light , , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

//
// Closest hit material
//
rtDeclareVariable(float3,  Ks, , );
rtDeclareVariable(float3,  Kd, , );
rtDeclareVariable(float3,  emitted, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PhotonPRD, hit_record, rtPayload, );


static __device__ __inline__ float2 rnd_from_uint2(uint2& prev)
{
	return make_float2(rnd(prev.x), rnd(prev.y));
}

RT_PROGRAM void ppass_closest_hit_transparent()
{
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	float3 hit_point = ray.origin + t_hit*ray.direction;
	float3 new_ray_dir;

	float3 U, V, W;
	createONB(ffnormal, U, V, W);
	sampleUnitHemisphere(rnd_from_uint2(hit_record.sample), U, V, W, new_ray_dir);

	optix::Ray new_ray( hit_point, new_ray_dir, ppass_and_gather_ray_type, scene_epsilon );
	rtTrace(top_object, new_ray, hit_record);
}

rtDeclareVariable(ShadowPRD, shadow_prd, rtPayload, );

RT_PROGRAM void gather_any_hit_glass()
{

}

rtDeclareVariable(HitPRD, hit_prd, rtPayload, );
rtBuffer<HitRecord, 3>           rtpass_output_buffer;

rtTextureSampler<float4, 2>      diffuse_map;
rtDeclareVariable(float, diffuse_map_scale, , );
rtDeclareVariable(float3, texcoord, attribute texcoord, );


static __device__ __inline__ float3 exp( const float3& x )
{
	return make_float3(exp(x.x), exp(x.y), exp(x.z));
}

rtTextureSampler<float4, 2> envmap;

RT_PROGRAM void rtpass_closest_hit_glass()
{
	float refraction_index = 1.4f;
	float fresnel_exponent = 3.0f;
	float fresnel_minimum = 0.1f;
	float fresnel_maximum = 1.0f;
	float importance_cutoff = 0.01f;
	float3 extinction = make_float3(1.0f, 1.0f, 1.0f);
	float3 extinction_constant= make_float3(log(extinction.x), log(extinction.y), log(extinction.z));
	float3 refraction_color = make_float3(1.0f, 1.0f, 1.0f);

	float3 direction    = ray.direction;
	float3 origin       = ray.origin;
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	float3 ffnormal     = faceforward( world_shading_normal, -direction, world_geometric_normal );
	float3 hit_point    = origin + t_hit*direction;
	double tHitStack = t_hit + 0.1 - 0.1; // Important, prevents compiler optimization on variable

	if( fmaxf( Kd ) > 0.0f )
	{
		float theta = atan2f( ray.direction.x, ray.direction.z );
		float phi   = M_PIf * 0.5f -  acosf( ray.direction.y );
		float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
		float v     = 0.5f * ( 1.0f + sin(phi) );
		float3 result = make_float3(tex2D(envmap, u, v));

		HitRecord rec = rtpass_output_buffer[make_uint3(launch_index.x,launch_index.y,0)];
		// We hit a diffuse surface; record hit and return
		rec.position = hit_point;
		rec.normal = ffnormal;
		rec.attenuated_Kd = Kd * hit_prd.attenuation;

		rec.flags = PPM_HIT;
		rec.attenuated_Kd *= make_float3(
				tex2D(diffuse_map, texcoord.x * diffuse_map_scale, texcoord.y * diffuse_map_scale)) * 100;
		rec.attenuated_Kd += result*2;
		//rtPrintf("%f %f %f\n", result.x, result.y, result.z);
		//rtPrintf("%f %f %f\n", rec.attenuated_Kd.x, rec.attenuated_Kd.y, rec.attenuated_Kd.z);
		rtpass_output_buffer[make_uint3(launch_index.x,launch_index.y,0)] = rec;
	}
	hit_prd.lastTHit = tHitStack;
}

RT_PROGRAM void any_hit_glass_rt()
{
	//rtIgnoreIntersection();
}

RT_PROGRAM void any_hit_glass_ph()
{
	float3 color = make_float3(tex2D(diffuse_map, texcoord.x*diffuse_map_scale, texcoord.y*diffuse_map_scale));
	//rtPrintf("%f %f %f\n", color.x, color.y, color.z);
	hit_record.energy*=color*5;
	rtIgnoreIntersection();
}