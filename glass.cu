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
rtBuffer<HitRecord, 2>           rtpass_output_buffer;

rtTextureSampler<float4, 2>      diffuse_map;
rtDeclareVariable(float, diffuse_map_scale, , );
rtDeclareVariable(float3, texcoord, attribute texcoord, );


static __device__ __inline__ float3 exp( const float3& x )
{
	return make_float3(exp(x.x), exp(x.y), exp(x.z));
}

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
		HitRecord rec = rtpass_output_buffer[launch_index];;
		float reflection = 1.0f;
		float3 result = make_float3(0.0f);

		float3 beer_attenuation;
		if(dot(world_shading_normal, ray.direction) > 0){
			// Beer's law attenuation
			beer_attenuation = exp(extinction_constant * 0.01);
		} else {
			beer_attenuation = exp(extinction_constant * 0.01);
		}

		float3 t;                                                            // transmission direction
		if ( refract(t, direction, world_shading_normal, refraction_index) )
		{
			// check for external or internal reflection
			float cos_theta = dot(direction, world_shading_normal);
			if (cos_theta < 0.0f)
				cos_theta = -cos_theta;
			else
				cos_theta = dot(t, world_shading_normal);

			reflection = fresnel_schlick(cos_theta, fresnel_exponent, fresnel_minimum, fresnel_maximum);

			optix::Ray ray(hit_point, t, radiance_in_participating_medium, scene_epsilon);
			HitPRD refr_prd;
			refr_prd.ray_depth = hit_prd.ray_depth + 1;
			refr_prd.attenuation = hit_prd.attenuation;

			rtTrace(top_object, ray, refr_prd);
			result += (1.0f - reflection) * refraction_color * refr_prd.attenuation;
		}

		// We hit a diffuse surface; record hit and return
		rec.position = hit_point;
		rec.normal = ffnormal;
		rec.attenuated_Kd = (Kd + result) * hit_prd.attenuation;

		rec.flags = PPM_HIT;
		rec.attenuated_Kd *= make_float3(
				tex2D(diffuse_map, texcoord.x * diffuse_map_scale, texcoord.y * diffuse_map_scale));
		//rtPrintf("%f %f %f\n", rec.attenuated_Kd.x, rec.attenuated_Kd.y, rec.attenuated_Kd.z);
		rtpass_output_buffer[launch_index] = rec;
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