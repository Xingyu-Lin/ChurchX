#include "ParticipatingMedium.h"
#include "ppm.h"
#include "SampleScene.h"


bool ParticipatingMedium::m_optixMaterialIsCreated = false;
optix::Material ParticipatingMedium::m_optixMaterial;

ParticipatingMedium::ParticipatingMedium(float sigma_s, float sigma_a)
	: m_sigma_a(sigma_a), m_sigma_s(sigma_s)
{
}

optix::Material ParticipatingMedium::getOptixMaterial(optix::Context & context, const std::string ptxpath)
{
	if (!m_optixMaterialIsCreated)
	{
		optix::Program radianceProgram = context->createProgramFromPTXFile(ptxpath, "closestHitRadiance");
		optix::Program photonProgram = context->createProgramFromPTXFile(ptxpath, "closestHitPhoton");
		//optix::Program transmissionProgram = context->createProgramFromPTXFile( "ParticipatingMedium.cu.ptx", "radianceTransmission");

		m_optixMaterial = context->createMaterial();
		m_optixMaterial->setClosestHitProgram(RayTypes::rtpass_ray_type, radianceProgram);
		m_optixMaterial->setClosestHitProgram(RayTypes::radiance_in_participating_medium, radianceProgram);
		m_optixMaterial->setClosestHitProgram(RayTypes::ppass_and_gather_ray_type, photonProgram);
		m_optixMaterial->setClosestHitProgram(RayTypes::photon_in_participating_medium, photonProgram);

		this->registerMaterialWithShadowProgram(context, m_optixMaterial);

		m_optixMaterialIsCreated = true;
	}

	return m_optixMaterial;
}

/*
// Register any material-dependent values to be available in the optix program.
*/

void ParticipatingMedium::registerGeometryInstanceValues(optix::GeometryInstance & instance)
{
	instance["participatingMedium"]->setUint(1);
	instance["sigma_a"]->setFloat(m_sigma_a);
	instance["sigma_s"]->setFloat(m_sigma_s);

}

void ParticipatingMedium::registerMaterialWithShadowProgram(optix::Context & context, optix::Material & material)
{
	//if (!m_hasLoadedOptixAnyHitProgram)
	//{
	//	m_optixAnyHitProgram = context->createProgramFromPTXFile("DirectRadianceEstimation.cu.ptx", "gatherAnyHitOnNonEmitter");
	//	m_hasLoadedOptixAnyHitProgram = true;
	//}
	//material->setAnyHitProgram(shadow_ray_type, m_optixAnyHitProgram);
	//TODO:
}