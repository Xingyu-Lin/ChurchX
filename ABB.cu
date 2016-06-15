/*
* Copyright (c) 2013 Opposite Renderer
* For the full copyright and license information, please view the LICENSE.txt
* file that was distributed with this source code.
*/

#include <optix_world.h>
#include "ppm.h"

using namespace optix;

rtDeclareVariable(float3, cuboidMin, , );
rtDeclareVariable(float3, cuboidMax, , );
rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, );
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );


static __device__ float3 boxnormal(float t)
{
	float3 t0 = (cuboidMin - ray.origin)/ray.direction;
	float3 t1 = (cuboidMax - ray.origin)/ray.direction;
	float3 neg = make_float3(t==t0.x?1:0, t==t0.y?1:0, t==t0.z?1:0);
	float3 pos = make_float3(t==t1.x?1:0, t==t1.y?1:0, t==t1.z?1:0);
	return pos-neg;
}

RT_PROGRAM void intersect(int primIdx)
{
	float3 origin = ray.origin;
	if (cuboidMin.x+0.001< origin.x && origin.x< cuboidMax.x-0.001 &&
		cuboidMin.y+0.001< origin.y && origin.y< cuboidMax.y-0.001 &&
		cuboidMin.z+0.001< origin.z && origin.z< cuboidMax.z-0.001 &&
	 (ray.ray_type == ppass_and_gather_ray_type || ray.ray_type == rtpass_ray_type)
	)
	{
		if (rtPotentialIntersection(0.00011f))
		{
			geometricNormal = -ray.direction;
			shadingNormal = geometricNormal;
			rtReportIntersection(0);
		}
		return;
	}

	float3 t0 = (cuboidMin - ray.origin)/ray.direction;
	float3 t1 = (cuboidMax - ray.origin)/ray.direction;
	float3 Near = fminf(t0, t1);
	float3 Far = fmaxf(t0, t1);
	float tmin = fmaxf( Near );
	float tmax = fminf( Far );

	if(tmin <= tmax) {
		bool check_second = true;
		if( rtPotentialIntersection( tmin ) ) {
			shadingNormal = geometricNormal = boxnormal( tmin );
			if(rtReportIntersection(0))
				check_second = false;
		}
		if(check_second) {
			if( rtPotentialIntersection( tmax ) ) {
				shadingNormal = geometricNormal = boxnormal( tmax );
				rtReportIntersection(0);
			}
		}
	}
}

RT_PROGRAM void boundingBox(int, float result[6])
{
	result[0] = cuboidMin.x;
	result[1] = cuboidMin.y;
	result[2] = cuboidMin.z;
	result[3] = cuboidMax.x;
	result[4] = cuboidMax.y;
	result[5] = cuboidMax.z;
}
