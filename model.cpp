#include "model.h"
#include <optixu/optixu_math_stream_namespace.h>

#include <OptiXMesh.h>

#include <iostream>
#include <cstdlib>
#include <SampleScene.h>
#include <memory.h>

using namespace optix;

Model::Model(std::string &objfilename,const optix::Material matl, AccelDescriptor& accel_desc, optix::TextureSampler projectedTexSamp, optix::Program intersectProgram,
	Program meshProgram, optix::Context context, optix::GeometryGroup inGG, optix::float3 translate, optix::float3 scale, float rotateRadius, optix::float3 rotateAxis) : m_geom_group(inGG), m_this_owns_geom_group(false)
{
	static bool printedPermissions = false;
	if (!printedPermissions) {
		if (objfilename.find("Fish_OBJ_PPM") != std::string::npos) {
			std::cout << "\nModels and textures copyright Toru Miyazawa, Toucan Corporation. Used by permission.\n";
			std::cout << "http://toucan.web.infoseek.co.jp/3DCG/3ds/FishModelsE.html\n\n";
			printedPermissions = true;
		}
	}

	//m_species = FishMonger_t::FindSpecies(objfilename);

	// std::cerr << "Found name: " << m_species->name << '\n';
	GeometryInstance GI;
	if (m_geom_group.get() == 0) {
		std::cerr << "Loading " << objfilename << '\n';
		m_this_owns_geom_group = true;

		m_geom_group = context->createGeometryGroup();
		OptiXMesh FishLoader(context, m_geom_group, matl, accel_desc);

		float m[16] = {
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
		};
		Matrix4x4 Rot(m);
		Matrix4x4 XForm = Rot;
		XForm = Matrix4x4::scale(scale) * XForm;
		XForm = Matrix4x4::rotate(rotateRadius, rotateAxis) * XForm;
		XForm = Matrix4x4::translate(translate) * XForm;


		FishLoader.setLoadingTransform(XForm);
		FishLoader.setDefaultIntersectionProgram(intersectProgram);
		FishLoader.setDefaultBoundingBoxProgram(meshProgram);

		FishLoader.loadBegin_Geometry(objfilename);
		FishLoader.loadFinish_Materials();

		m_aabb = FishLoader.getSceneBBox();

		GI = m_geom_group->getChild(0);
		GI["diffuse_map_scale"]->setFloat(1.0f);
		if (objfilename == "pod")
			GI["emitted"]->setFloat(0.1f, 0.0f, 0.0f);
		GI["Kd"]->setFloat(0.4, 0.4, 0.4);
		if (objfilename == "pod") {
			GI["Ks"]->setFloat(1.0, 1.0, 1.0);
		}
		GI["Ka"]->setFloat(0.5, 0.5, 0.5);
		GI["grid_color"]->setFloat(0.5f, 0.5f, 0.5f);
		optix::Buffer buff = GI["diffuse_map"]->getBuffer();
	}
	else {
		GI = m_geom_group->getChild(0);
	}

	m_geom = GI->getGeometry();
	m_vert_buff = m_geom["vertex_buffer"]->getBuffer();
	m_vert_buff->getSize(m_num_verts); // Query number of vertices in the buffer

	m_norm_buff = m_geom["normal_buffer"]->getBuffer();
	m_norm_buff->getSize(m_num_norms); // Query number of normals in the buffer, which doesn't match the number of vertices

	m_vindices_buff = m_geom["vindex_buffer"]->getBuffer();
	m_vindices_buff->getSize(m_num_tris);

	m_nindices_buff = m_geom["nindex_buffer"]->getBuffer();

	m_tran = context->createTransform();
	m_tran->setChild(m_geom_group);
}
