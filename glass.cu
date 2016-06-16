#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "ppm.h"
#include "path_tracer.h"
#include "random.h"

using namespace optix;

rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(float3, emitted, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PhotonPRD, hit_record, rtPayload, );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(float, scene_epsilon, , );
rtBuffer<PhotonRecord, 1>        ppass_output_buffer;

static __device__ __inline__ float2 rnd_from_uint2(uint2& prev)
{
	return make_float2(rnd(prev.x), rnd(prev.y));
}

RT_PROGRAM void ppass_closest_hit_transparent()
{
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	float3 hit_point = ray.origin;
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
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

RT_PROGRAM void rtpass_closest_hit_glass()
{
	float3 hit_point = ray.origin + t_hit*ray.direction;
	float3 new_ray_dir = ray.direction;
	hit_record.energy = hit_record.energy;
	optix::Ray new_ray(hit_point, new_ray_dir, rtpass_ray_type, scene_epsilon);
	rtTrace(top_object, new_ray, hit_record);
}