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

#ifndef __samples_util_optix_mesh_impl_h__
#define __samples_util_optix_mesh_impl_h__

#include "AccelDescriptor.h"
#include "MeshBase.h"
#include "OptiXMeshClasses.h"

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include <string>


//-----------------------------------------------------------------------------
// 
//  OptiXMeshImpl class declaration 
//
//-----------------------------------------------------------------------------
class OptiXMeshImpl : public MeshBase
{
public:

  typedef MeshBase Base;

  OptiXMeshImpl( optix::Context context,              // Context for RT object creation
                 optix::GeometryGroup geometrygroup,  // Empty geom group to hold model
                 const AccelDescriptor& accel_desc ); // Parameters of the acceleration structure

  OptiXMeshImpl( optix::Context context,              // Context for RT object creation
                 optix::GeometryGroup geometrygroup,  // Empty geom group to hold model
                 optix::Material default_material,    // Default OptiX material to assign
                 const AccelDescriptor& accel_desc ); // Parameters of the acceleration structure

  virtual ~OptiXMeshImpl();

  /// Not intended for use; throws a MeshException to override base class method.
  virtual void loadModel( const std::string& filename );

  void beginLoad( const std::string& filename );
  void finalizeLoad();

  void setDefaultIntersectionProgram( optix::Program inx_program );
  optix::Program getDefaultIntersectionProgram() const;
  void setDefaultBoundingBoxProgram( optix::Program bbox_program );
  optix::Program getDefaultBoundingBoxProgram() const;
  void setDefaultOptiXMaterial( optix::Material material );
  optix::Material getDefaultOptiXMaterial() const;
  void setDefaultMaterialParams( const MeshMaterialParams& params );
  const MeshMaterialParams& getDefaultMaterialParams() const;

  OptiXMeshClasses::GroupGeometryInfo& getGroupGeometryInfo( const std::string& group_name );
  const OptiXMeshClasses::GroupGeometryInfo& getGroupGeometryInfo( const std::string& group_name ) const;

  void setOptiXMaterial( int i, optix::Material material );
  const optix::Material& getOptiXMaterial( int i ) const;

  optix::Aabb getSceneBBox() const;
  optix::Buffer getLightBuffer() const;

  const optix::Matrix4x4& getLoadingTransform() const;
  void setLoadingTransform( const optix::Matrix4x4& transform );

  virtual void addMaterial();
  int getGroupMaterialNumber(const MeshGroup& group) const;
  void setOptixInstanceMatParams( optix::GeometryInstance gi,
                                  const MeshMaterialParams& params ) const;

protected:

  virtual void preProcess();
  virtual void allocateData();
  virtual void startWritingData();
  virtual void postProcess();
  virtual void finishWritingData();

private:

  typedef std::map<std::string, OptiXMeshClasses::GroupGeometryInfo> GroupGeometryInfoMap;

  void createGeometries();
  
  void mapPoolBuffers();
  void unmapPoolBuffers();

  void mapGroupBuffers();
  void unmapGroupBuffers();

  void mapAllBuffers();
  void unmapAllBuffers();

  void processGeometriesAfterLoad();
  void createGeoInstancesAndOptixGroup();
  void createLightBuffer();
  void transformVerticesAndNormals();
  
  void initDefaultOptiXMaterial();
  void initDefaultIntersectProgram();
  void initDefaultBBoxProgram();
  
  std::string getTextureMapPath( const std::string& filename ) const;

  optix::Buffer getGeoVindexBuffer( optix::Geometry geo );
  void setGeoVindexBuffer( optix::Geometry geo, optix::Buffer buf );
  optix::Buffer getGeoNindexBuffer( optix::Geometry geo );
  void setGeoNindexBuffer( optix::Geometry geo, optix::Buffer buf );
  optix::Buffer getGeoTindexBuffer( optix::Geometry geo );
  void setGeoTindexBuffer( optix::Geometry geo, optix::Buffer buf );
  optix::Buffer getGeoMaterialBuffer( optix::Geometry geo );
  void setGeoMaterialBuffer( optix::Geometry geo, optix::Buffer buf );

  optix::Context         m_context;
  optix::GeometryGroup   m_geometrygroup;
  optix::Buffer          m_vbuffer;
  optix::Buffer          m_nbuffer;
  optix::Buffer          m_tbuffer;

  MeshMaterialParams     m_default_material_params;
  optix::Material        m_default_optix_material;
  optix::Program         m_default_intersect_program;
  optix::Program         m_default_bbox_program;
  
  optix::Buffer          m_light_buffer;
  
  AccelDescriptor        m_accel_desc;

  optix::Aabb            m_aabb;
  
  std::vector<optix::Material> m_optix_materials;
  GroupGeometryInfoMap         m_group_infos;
  
  optix::Matrix4x4       m_loading_transform;

  bool m_materials_provided;

  bool m_currently_loading;

  // Implementation functors
  struct InitGroupsDefaultIntersectFunctor;
  struct CreateGeometriesFunctor;
  struct MapGroupBuffersFunctor;
  struct UnmapGroupBuffersFunctor;
  struct ProcessGeometriesAfterLoadFunctor;
  struct CreateGeometryInstancesFunctor;
  struct CreateLightBufferFunctor;
};


#endif // __samples_util_optix_mesh_impl_h__
