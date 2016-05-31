/*
* Copyright (c) 2013 Opposite Renderer
* For the full copyright and license information, please view the LICENSE.txt
* file that was distributed with this source code.
*/

#pragma once
#include <optixu/optixpp_namespace.h>
class ParticipatingMedium
{
public:
	ParticipatingMedium(float sigma_s, float sigma_a);
	optix::Material getOptixMaterial(optix::Context & context, const std::string ptxpath);
	void registerGeometryInstanceValues(optix::GeometryInstance & instance);
	static void registerMaterialWithShadowProgram(optix::Context & context, optix::Material & material);
private:
	//float indexOfRefraction;
	static bool m_optixMaterialIsCreated;
	static optix::Material m_optixMaterial;
	float m_sigma_s;
	float m_sigma_a;
	//static bool m_hasLoadedOptixAnyHitProgram;
	static optix::Program m_optixAnyHitProgram;
};