#pragma once

#include "AccelDescriptor.h"

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

class Model
{
public:
	Model(){};
	Model(std::string &objfilename, optix::Material matl, AccelDescriptor& accel_desc, optix::TextureSampler projectedTexSamp,
		optix::Program intersectProgram, optix::Program meshProgram, optix::Context context, optix::GeometryGroup inGG,
		optix::float3 translate, optix::float3 scale, float rotateRadius, optix::float3 rotateAxis);

	//void initAnimation(optix::Aabb TargetBox, float fishFrac, bool deterministic);
	void updateGeometry(optix::Aabb TargetBox);
	void scale(float times);
	optix::float3 target_pos() const { return m_target_pos; }
	void target_pos(optix::float3 val) { m_target_pos = val; }
	bool owns_geom_group() const { return m_this_owns_geom_group; }

	optix::Transform     m_tran;
	optix::GeometryGroup m_geom_group;

private:
	//void swimVertex(const float phaseDeg, optix::float3& P, optix::float3& N);

	//void swimPoint(optix::float3 &P, const float phaseDeg);

	void updatePos(optix::Aabb TargetBox);
	void updatePosCircle(optix::Aabb TargetBox);

	//std::vector<std::vector<optix::float3> > m_animated_points;
	//std::vector<std::vector<optix::float3> > m_animated_normals;

	optix::Geometry      m_geom;
	optix::Buffer        m_vert_buff;
	optix::Buffer        m_norm_buff;
	optix::Buffer        m_vindices_buff;
	optix::Buffer        m_nindices_buff;
	optix::Aabb          m_aabb;
	optix::float3        m_pos, m_vel, m_target_pos;
	int                  m_phase_num;
	int                  m_frames_this_target; // Frames since m_target_pos was chosen
	RTsize               m_num_verts;
	RTsize               m_num_norms;
	RTsize               m_num_tris;
	bool                 m_this_owns_geom_group; // True if this Fish is responsible for updating the shared GG for this species each frame
	bool                 m_swim_deterministically;

	const static int ANIM_STEPS = 31;
};
