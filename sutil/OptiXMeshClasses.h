#ifndef __samples_util_optix_mesh_classes_h__
#define __samples_util_optix_mesh_classes_h__

#include <optixpp_namespace.h>
#include <optixu_math_namespace.h>


namespace OptiXMeshClasses {

struct GroupGeometryInfo {
  optix::Program intersect_program;
  optix::Program bbox_program;

  optix::Geometry         m_geometry;
  optix::GeometryInstance m_geometry_instance;

  unsigned int* mbuffer_data; // material buffer pointer
};

} // namespace OptiXMeshClasses

#endif // __samples_util_optix_mesh_classes_h__
