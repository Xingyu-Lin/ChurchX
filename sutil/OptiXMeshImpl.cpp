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

#include "AccelDescriptor.h"
#include "MeshBase.h"
#include "OptiXMeshClasses.h"
#include "OptiXMeshImpl.h"

#include <commonStructs.h>
#include <ImageLoader.h>

#include <optixu/optixu.h>
#include <optixu/optixu_math_namespace.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string.h>
#include <vector>


using namespace OptiXMeshClasses;
using namespace optix;

//------------------------------------------------------------------------------
// 
//  OptiXMeshImpl class definition 
//
//------------------------------------------------------------------------------
OptiXMeshImpl::OptiXMeshImpl( Context context,
                              GeometryGroup geometrygroup,
                              const AccelDescriptor& accel_desc )
: Base(),
  m_context( context ),
  m_geometrygroup( geometrygroup ),
  m_vbuffer( 0 ),
  m_nbuffer( 0 ),
  m_tbuffer( 0 ),
  m_default_optix_material( 0 ),
  m_default_intersect_program( 0 ),
  m_default_bbox_program( 0 ),
  m_accel_desc( accel_desc ),
  m_aabb(),
  m_loading_transform(optix::Matrix4x4::identity()),
  m_materials_provided( false ),
  m_currently_loading( false )
{ }

OptiXMeshImpl::OptiXMeshImpl( Context context,
                              GeometryGroup geometrygroup,
                              optix::Material default_material,
                              const AccelDescriptor& accel_desc )
: Base(),
  m_context( context ),
  m_geometrygroup( geometrygroup ),
  m_vbuffer( 0 ),
  m_nbuffer( 0 ),
  m_tbuffer( 0 ),
  m_default_optix_material( default_material ),
  m_default_intersect_program( 0 ),
  m_default_bbox_program( 0 ),
  m_accel_desc( accel_desc ),
  m_aabb(),
  m_loading_transform(optix::Matrix4x4::identity()),
  m_materials_provided( false ),
  m_currently_loading( false )
{ }

OptiXMeshImpl::~OptiXMeshImpl() { } // makes sure CRT objects are destroyed on the correct heap


void OptiXMeshImpl::loadModel( const std::string& filename )
{
  throw MeshException("Use beginLoad() and finalizeLoad() instead of loadModel()");
}

void OptiXMeshImpl::beginLoad( const std::string& filename )
{
  if( m_currently_loading )
    throw MeshException("Loading has already begun.");
  m_currently_loading = true;

  Base::loadModel( filename );
}

void OptiXMeshImpl::finalizeLoad()
{
  if( !m_currently_loading )
    throw MeshException("finalizeLoad() can only be called after a call to beginLoad()");

  mapAllBuffers();
  processGeometriesAfterLoad();
  createGeoInstancesAndOptixGroup();
  createLightBuffer();
  unmapAllBuffers();

  m_currently_loading = false;
}


void OptiXMeshImpl::setDefaultIntersectionProgram( optix::Program inx_program ) {
  m_default_intersect_program = inx_program;
}
optix::Program OptiXMeshImpl::getDefaultIntersectionProgram() const {
  return m_default_intersect_program;
}

void OptiXMeshImpl::setDefaultBoundingBoxProgram( optix::Program bbox_program ) {
  m_default_bbox_program = bbox_program;
}
optix::Program OptiXMeshImpl::getDefaultBoundingBoxProgram() const {
  return m_default_bbox_program;
}

void OptiXMeshImpl::setDefaultOptiXMaterial( optix::Material material ) {
  m_default_optix_material = material;
}
optix::Material OptiXMeshImpl::getDefaultOptiXMaterial() const {
  return m_default_optix_material;
}

void OptiXMeshImpl::setDefaultMaterialParams( const MeshMaterialParams& params ) {
  m_default_material_params = params;
}
const MeshMaterialParams& OptiXMeshImpl::getDefaultMaterialParams() const {
  return m_default_material_params;
}


GroupGeometryInfo& OptiXMeshImpl::getGroupGeometryInfo( const std::string& group_name ) {
  return m_group_infos[group_name];
}

const GroupGeometryInfo&
OptiXMeshImpl::getGroupGeometryInfo( const std::string& group_name ) const
{
  return m_group_infos.at(group_name);
    // There is no 'const' version of operator[] for maps, so we have to use
    // map::at()
}


void OptiXMeshImpl::setOptiXMaterial( int i, optix::Material material ) {
  m_optix_materials[i] = material;
}

const optix::Material& OptiXMeshImpl::getOptiXMaterial( int i ) const {
  return m_optix_materials[i];
}

optix::Aabb OptiXMeshImpl::getSceneBBox() const { return m_aabb; }

optix::Buffer OptiXMeshImpl::getLightBuffer() const { return m_light_buffer; }


const optix::Matrix4x4& OptiXMeshImpl::getLoadingTransform() const
{
  return m_loading_transform;
}

void OptiXMeshImpl::setLoadingTransform( const optix::Matrix4x4& transform )
{
  m_loading_transform = transform;
}


void OptiXMeshImpl::addMaterial() {
  Base::addMaterial();

  m_optix_materials.push_back( m_default_optix_material );
}

int OptiXMeshImpl::getGroupMaterialNumber(const MeshGroup& group) const
{
  return m_materials_provided ? group.material_number : 0;
}

    
void OptiXMeshImpl::preProcess()
{
  if( !m_default_optix_material ) initDefaultOptiXMaterial();
  if( !m_default_intersect_program ) initDefaultIntersectProgram();
  if( !m_default_bbox_program ) initDefaultBBoxProgram();
}


void OptiXMeshImpl::allocateData()
{
  // Create vertex, normal, and texture_coordinate buffers
  // (colors are not used)
  m_vbuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, getNumVertices() );
  m_nbuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, getNumNormals() );
  m_tbuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2,
                                       getNumTextureCoordinates() );

  // Signal to base class loading code that we don't use the color data...
  setColorData( 0 );

  // ...  and that buffers are compacted
  setVertexStride( 0 );
  setNormalStride( 0 );
  setColorStride( 0 );
  setTextureCoordinateStride( 0 );

  createGeometries();
}

void OptiXMeshImpl::startWritingData()
{
  mapAllBuffers();
}


struct OptiXMeshImpl::InitGroupsDefaultIntersectFunctor {
  InitGroupsDefaultIntersectFunctor( OptiXMeshImpl& mesh ) : m_mesh(mesh) { }

  void operator()( MeshGroup& group ) {
    GroupGeometryInfo& group_info = m_mesh.getGroupGeometryInfo( group.name );
    group_info.intersect_program = m_mesh.m_default_intersect_program;
    group_info.bbox_program = m_mesh.m_default_bbox_program;
  }

private:
    OptiXMeshImpl& m_mesh;
};


void OptiXMeshImpl::postProcess()
{
  // Attach a default intersection and bounding box program to each model group
  forEachGroup( InitGroupsDefaultIntersectFunctor(*this) );

  // Prep provided material params with an accompanying default Phong Material
  // program, or add one params/program pair if no material params were
  // loaded
  m_materials_provided = (getMaterialCount() > 0);
  if( m_materials_provided ) {
    // Prep m_optix_materials and m_group_infos with defaults
    for( size_t i = 0; i < getMaterialCount(); ++i ) {
      m_optix_materials.push_back( m_default_optix_material );
    }
  }
  else {
    addMaterial();
  }

  transformVerticesAndNormals();
}

void OptiXMeshImpl::finishWritingData()
{
  unmapAllBuffers();
}


struct OptiXMeshImpl::CreateGeometriesFunctor {

  CreateGeometriesFunctor( OptiXMeshImpl& mesh ) : m_mesh( mesh ) { }
  
  void operator()( MeshGroup& group ) {

    assert( group.num_triangles > 0 );

    group.material_number = 0;
    GroupGeometryInfo& group_geo_info = m_mesh.getGroupGeometryInfo( group.name );

    optix::Geometry geo = m_mesh.m_context->createGeometry();
    group_geo_info.m_geometry = geo;
        
    // Create vertex index buffers
    optix::Buffer vindex_buffer
      = m_mesh.m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3,
                                        group.num_triangles );
    optix::Buffer tindex_buffer
      = m_mesh.m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3,
                                        group.num_triangles );
    optix::Buffer nindex_buffer
      = m_mesh.m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3,
                                        group.num_triangles );
    optix::Buffer mbuffer
      = m_mesh.m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT,
                                        group.num_triangles );
    
    group_geo_info.mbuffer_data
      = static_cast<unsigned int*>( mbuffer->map() );
    for( int j = 0; j < group.num_triangles; ++j ) {
      group_geo_info.mbuffer_data[j] = 0; // See above TODO
                                          //   What TODO? HAS LORE BEEN LOST??
    }
    mbuffer->unmap();
    group_geo_info.mbuffer_data = 0;

    geo["vertex_buffer"]->setBuffer( m_mesh.m_vbuffer );
    m_mesh.setGeoVindexBuffer( geo, vindex_buffer );

    geo["normal_buffer"]->setBuffer( m_mesh.m_nbuffer );
    m_mesh.setGeoNindexBuffer( geo, nindex_buffer );

    geo["texcoord_buffer"]->setBuffer( m_mesh.m_tbuffer );
    m_mesh.setGeoTindexBuffer( geo, tindex_buffer );

    m_mesh.setGeoMaterialBuffer( geo, mbuffer );
  }

private:
  OptiXMeshImpl& m_mesh;
};

void OptiXMeshImpl::createGeometries()
{
  forEachGroup( CreateGeometriesFunctor(*this) );
}


void OptiXMeshImpl::mapPoolBuffers()
{
  setVertexData( static_cast<float*>( m_vbuffer->map() ) );
  setNormalData( static_cast<float*>( m_nbuffer->map() ) );
  setTextureCoordinateData( static_cast<float*>( m_tbuffer->map() ) );
}

void OptiXMeshImpl::unmapPoolBuffers()
{
  setTextureCoordinateData( 0 );
  m_tbuffer->unmap();
  
  setNormalData( 0 );
  m_nbuffer->unmap();

  setVertexData( 0 );
  m_vbuffer->unmap();
}


struct OptiXMeshImpl::MapGroupBuffersFunctor
{
  OptiXMeshImpl& m_mesh;
  MapGroupBuffersFunctor( OptiXMeshImpl& mesh ) : m_mesh( mesh ) { }

  void operator()( MeshGroup& group ) {
    GroupGeometryInfo& group_geo_info = m_mesh.getGroupGeometryInfo( group.name );
    optix::Geometry geo = group_geo_info.m_geometry;

    group.vertex_indices
      = static_cast<int*>( m_mesh.getGeoVindexBuffer( geo )->map() );
    group.normal_indices
      = static_cast<int*>( m_mesh.getGeoNindexBuffer( geo )->map() );
    group.texture_coordinate_indices
      = static_cast<int*>( m_mesh.getGeoTindexBuffer( geo )->map() );
    // (group.color_indices is not used)

    group_geo_info.mbuffer_data
      = static_cast<unsigned int*>( m_mesh.getGeoMaterialBuffer( geo )->map() );
  }
};

void OptiXMeshImpl::mapGroupBuffers()
{
  forEachGroup( MapGroupBuffersFunctor(*this) );
}


struct OptiXMeshImpl::UnmapGroupBuffersFunctor 
{
  UnmapGroupBuffersFunctor( OptiXMeshImpl& mesh ) : m_mesh( mesh ) { }

  void operator()( MeshGroup& group ) {
    GroupGeometryInfo& group_geo_info = m_mesh.getGroupGeometryInfo( group.name );
    optix::Geometry geo = group_geo_info.m_geometry;

    m_mesh.getGeoVindexBuffer( geo )->unmap();
    group.vertex_indices = 0;

    m_mesh.getGeoNindexBuffer( geo )->unmap();
    group.normal_indices = 0;

    m_mesh.getGeoTindexBuffer( geo )->unmap();
    group.texture_coordinate_indices = 0;
    
    m_mesh.getGeoMaterialBuffer( geo )->unmap();
  }

private:
  OptiXMeshImpl& m_mesh;
};

void OptiXMeshImpl::unmapGroupBuffers()
{
  forEachGroup( UnmapGroupBuffersFunctor(*this) );
}


void OptiXMeshImpl::mapAllBuffers()
{
  mapPoolBuffers();
  mapGroupBuffers();
}

void OptiXMeshImpl::unmapAllBuffers()
{
  unmapGroupBuffers();
  unmapPoolBuffers();
}


// This functor assumes that all buffers for the GroupGeometryInfo associated
// with the MeshGroup passed to operator() are already mapped.  If m_large_geom
// is true, The optix::Geometry assigned to the GroupGeometryInfo will be
// replaced with a new one, so the old buffers will by unmapped and destroyed,
// and the newly created ones for the new one will be mapped, so that any
// following code will work that also assumes the buffers are mapped.
struct OptiXMeshImpl::ProcessGeometriesAfterLoadFunctor {
  
  ProcessGeometriesAfterLoadFunctor( OptiXMeshImpl& mesh ) : m_mesh( mesh ) { }

  void operator()( MeshGroup& group ) {
    
    OptiXMeshClasses::GroupGeometryInfo& group_geo_info
      = m_mesh.getGroupGeometryInfo( group.name );
    unsigned int num_triangles = static_cast<unsigned int>( group.num_triangles );
    assert( num_triangles > 0 );

    if( m_mesh.m_accel_desc.large_mesh ) {
      if( m_mesh.m_accel_desc.builder == std::string("Sbvh") || m_mesh.m_accel_desc.builder == std::string("Trbvh") || m_mesh.m_accel_desc.builder == std::string("KdTree") ) {
        // Splitting doesn't work with large_mesh, I think.
        m_mesh.m_accel_desc.builder = "MedianBvh";
        m_mesh.m_accel_desc.traverser = "Bvh";
      }

      RTgeometry geometry;

      unsigned int usePTX32InHost64 = 0;
      rtuCreateClusteredMeshExt( m_mesh.m_context->get(),
                                 usePTX32InHost64,
                                 &geometry, 
                                 (unsigned int)m_mesh.getNumVertices(),
                                 m_mesh.getVertexData(),
                                 (unsigned int)num_triangles,
                                 (const unsigned int*)group.vertex_indices,
                                 group_geo_info.mbuffer_data,
                                 m_mesh.m_nbuffer->get(),
                                 (const unsigned int*)group.normal_indices,
                                 m_mesh.m_tbuffer->get(), 
                                 (const unsigned int*)group.texture_coordinate_indices );
      
      optix::Geometry old_geo = group_geo_info.m_geometry;
      UnmapGroupBuffersFunctor old_geo_unmapper(m_mesh);
      old_geo_unmapper(group);
      rtBufferDestroy( m_mesh.getGeoVindexBuffer( old_geo )->get() );
      rtBufferDestroy( m_mesh.getGeoNindexBuffer( old_geo )->get() );
      rtBufferDestroy( m_mesh.getGeoTindexBuffer( old_geo )->get() );
      rtBufferDestroy( m_mesh.getGeoMaterialBuffer( old_geo )->get() );

      group_geo_info.m_geometry = optix::Geometry::take( geometry );
      MapGroupBuffersFunctor new_geo_mapper(m_mesh);
      new_geo_mapper(group);

      // (intersection and bounding box programs are attached by the
      // rtuCreateClusteredMeshExt call above)
    }
    else {
      optix::Geometry geo = group_geo_info.m_geometry;
      geo->setPrimitiveCount( num_triangles );
      geo->setIntersectionProgram( group_geo_info.intersect_program );
      geo->setBoundingBoxProgram( group_geo_info.bbox_program );
    }
    
  }

private:
  OptiXMeshImpl& m_mesh;
};

void OptiXMeshImpl::processGeometriesAfterLoad()
{
  forEachGroup( ProcessGeometriesAfterLoadFunctor(*this) );
}


struct OptiXMeshImpl::CreateGeometryInstancesFunctor
{
  OptiXMeshImpl& m_mesh;
  std::vector<GeometryInstance>& m_instances;

  CreateGeometryInstancesFunctor( OptiXMeshImpl& mesh,
                                  std::vector<GeometryInstance>& instances )
    : m_mesh( mesh ), m_instances( instances )
  { }

  void operator()( MeshGroup& group ) {

    GroupGeometryInfo& group_geo_info = m_mesh.getGroupGeometryInfo( group.name );
    optix::Geometry geo = group_geo_info.m_geometry;

    int material_number = m_mesh.getGroupMaterialNumber( group );
    const MeshMaterialParams& material_params = m_mesh.getMeshMaterialParams( material_number );
    optix::Material material = m_mesh.getOptiXMaterial( material_number );

    // Create the geom instance to hold mesh and material params
    GeometryInstance instance = m_mesh.m_context->createGeometryInstance( geo, &material, &material + 1 );
    m_mesh.setOptixInstanceMatParams( instance, material_params );
    m_instances.push_back( instance );
  }
};

void OptiXMeshImpl::createGeoInstancesAndOptixGroup()
{
  std::vector<GeometryInstance> instances;
  forEachGroup( CreateGeometryInstancesFunctor( *this, instances ) );

  // If we've already built an AS for this GG, just go with it. Otherwise, set it up.
  optix::Acceleration acceleration;
  acceleration = m_geometrygroup->getAcceleration();
  if(acceleration == 0) {
    acceleration = m_context->createAcceleration( m_accel_desc.builder.c_str(), m_accel_desc.traverser.c_str() );
    acceleration->setProperty( "refine", m_accel_desc.refine );
    acceleration->setProperty( "refit", m_accel_desc.refit );
    acceleration->setProperty( "vertex_buffer_name", "vertex_buffer" ); // Set these regardless of builder type. Ignored by some builders.
    acceleration->setProperty( "index_buffer_name", "vindex_buffer" );
    if( m_accel_desc.large_mesh )
      acceleration->setProperty( "leaf_size", "1" );
    m_geometrygroup->setAcceleration( acceleration );
  }

  acceleration->markDirty();

  for ( unsigned int i = 0; i < instances.size(); ++i )
    m_geometrygroup->addChild( instances[i] );
}


void OptiXMeshImpl::setOptixInstanceMatParams( GeometryInstance gi,
                                               const MeshMaterialParams& params ) const
{
  float3 Kd = *reinterpret_cast<const float3*>( params.diffuse );
  float3 Ka = *reinterpret_cast<const float3*>( params.ambient );
  float3 Ks = *reinterpret_cast<const float3*>( params.specular );

  // load textures relatively to OBJ main file
  std::string ambient_map_path  = getTextureMapPath( params.ambient_map.name );
  std::string diffuse_map_path  = getTextureMapPath( params.diffuse_map.name );
  std::string specular_map_path = getTextureMapPath( params.specular_map.name );

  gi[ "emissive"     ]->setFloat( *reinterpret_cast<const float3*>( params.emissive ) );
  gi[ "reflectivity" ]->setFloat( *reinterpret_cast<const float3*>( params.reflectivity ) );
  gi[ "phong_exp"    ]->setFloat( params.phong_exponent );
  gi[ "illum"        ]->setInt( params.shading_type );

  gi[ "ambient_map"  ]->setTextureSampler( loadTexture( m_context, ambient_map_path, Ka ) );
  gi[ "diffuse_map"  ]->setTextureSampler( loadTexture( m_context, diffuse_map_path, Kd ) );
  gi[ "specular_map" ]->setTextureSampler( loadTexture( m_context, specular_map_path, Ks ) );

  return;
}


struct OptiXMeshImpl::CreateLightBufferFunctor
{
  OptiXMeshImpl& m_mesh;
  std::vector<TriangleLight>& m_lights;

  CreateLightBufferFunctor( OptiXMeshImpl& mesh, std::vector<TriangleLight>& lights)
    : m_mesh( mesh ), m_lights( lights )
  { }

  void operator()( MeshGroup& group ) {

    size_t num_triangles = group.num_triangles;
    if ( num_triangles == 0 ) return;

    const MeshMaterialParams& mat = m_mesh.getMeshMaterialParams( m_mesh.getGroupMaterialNumber(group) );
    if ( (mat.emissive[0] + mat.emissive[1] + mat.emissive[2]) > 0.0f ) 
    {
      // extract necessary data
      for ( unsigned int j = 0; j < num_triangles; ++j ) 
      {
        // indices for vertex data
        int3 vindices = reinterpret_cast<int3*>(group.vertex_indices)[j];

        TriangleLight light;
        light.v1 = reinterpret_cast<float3*>(m_mesh.getVertexData())[vindices.x];
        light.v2 = reinterpret_cast<float3*>(m_mesh.getVertexData())[vindices.y];
        light.v3 = reinterpret_cast<float3*>(m_mesh.getVertexData())[vindices.z];

        // normal vector
        light.normal = normalize( cross( light.v2 - light.v3, light.v1 - light.v3 ) );

        light.emission = *reinterpret_cast<const float3*>( mat.emissive );

        m_lights.push_back(light);
      }
    }
  }
};


void OptiXMeshImpl::createLightBuffer()
{
  // create a buffer for the next-event estimation
  m_light_buffer = m_context->createBuffer( RT_BUFFER_INPUT );
  m_light_buffer->setFormat( RT_FORMAT_USER );
  m_light_buffer->setElementSize( sizeof( TriangleLight ) );

  // light sources
  std::vector<TriangleLight> lights;
  if ( getMaterialCount() > 0) {
    forEachGroup( CreateLightBufferFunctor( *this, lights ) );
  }

  // write to the buffer
  size_t num_lights = lights.size();
  m_light_buffer->setSize( 0 );
  if (num_lights != 0) {
    m_light_buffer->setSize( num_lights );
    memcpy( m_light_buffer->map(), &lights[0], num_lights * sizeof( TriangleLight ) );
    m_light_buffer->unmap();
  }
}


void OptiXMeshImpl::transformVerticesAndNormals()
{
  float3* vertices_f3 = reinterpret_cast<float3*>( getVertexData() );
  float3* normals_f3  = reinterpret_cast<float3*>( getNormalData() );

  optix::Aabb new_bbox;
  // Transform vertices
  int num_vertices = getNumVertices();
  for ( int i = 0; i < num_vertices; ++i )
  {
    float4 v4 = make_float4( vertices_f3[i], 1.0f );
    vertices_f3[i] = make_float3( m_loading_transform * v4 );
  }

  // Transform normals.
  int num_normals = getNumNormals();
  const Matrix4x4 norm_transform = m_loading_transform.inverse().transpose();
  for( int i = 0; i < num_normals; ++i )
  {
    float4 v4 = make_float4( normals_f3[i], 0.0f );
    normals_f3[i] = make_float3( norm_transform * v4 );
  }

  updateBBox();
  m_aabb.m_min = reinterpret_cast<const float3*>( getBBoxMin() )[0];
  m_aabb.m_max = reinterpret_cast<const float3*>( getBBoxMax() )[0];
}




void OptiXMeshImpl::initDefaultOptiXMaterial()
{
  std::string path = std::string(sutilSamplesPtxDir())
                   + "/cuda_compile_ptx_generated_obj_material.cu.ptx";

  Program closest_hit = m_context->createProgramFromPTXFile( path, "closest_hit_radiance" );
  Program any_hit     = m_context->createProgramFromPTXFile( path, "any_hit_shadow" );
  m_default_optix_material = m_context->createMaterial();
  m_default_optix_material->setClosestHitProgram( 0u, closest_hit );
  m_default_optix_material->setAnyHitProgram( 1u, any_hit ) ;
}


void OptiXMeshImpl::initDefaultIntersectProgram()
{
  std::string path = std::string(sutilSamplesPtxDir())
                     + "/cuda_compile_ptx_generated_triangle_mesh.cu.ptx";

  m_default_intersect_program
    = m_context->createProgramFromPTXFile( path, "mesh_intersect" );
}

void OptiXMeshImpl::initDefaultBBoxProgram()
{
  std::string path = std::string(sutilSamplesPtxDir())
                     + "/cuda_compile_ptx_generated_triangle_mesh.cu.ptx";

  m_default_bbox_program
    = m_context->createProgramFromPTXFile( path, "mesh_bounds" );
}

std::string OptiXMeshImpl::getTextureMapPath( const std::string& filename ) const
{
  if( filename.size() > 0 )
    return getPathName() + filename;
  else
    return "";
}


optix::Buffer OptiXMeshImpl::getGeoVindexBuffer( optix::Geometry geo )
{
  if( m_accel_desc.large_mesh )
    return geo["vert_idx_buffer"]->getBuffer();
  else
    return geo["vindex_buffer"]->getBuffer();
}

void OptiXMeshImpl::setGeoVindexBuffer( optix::Geometry geo, optix::Buffer buf )
{
  if( m_accel_desc.large_mesh )
    geo["vert_idx_buffer"]->setBuffer( buf );
  else
    geo["vindex_buffer"]->setBuffer( buf );
}

optix::Buffer OptiXMeshImpl::getGeoNindexBuffer( optix::Geometry geo )
{
  return geo["nindex_buffer"]->getBuffer();
}

void OptiXMeshImpl::setGeoNindexBuffer( optix::Geometry geo, optix::Buffer buf )
{
  geo["nindex_buffer"]->setBuffer( buf );
}

optix::Buffer OptiXMeshImpl::getGeoTindexBuffer( optix::Geometry geo )
{
  return geo["tindex_buffer"]->getBuffer();
}

void OptiXMeshImpl::setGeoTindexBuffer( optix::Geometry geo, optix::Buffer buf )
{
  geo["tindex_buffer"]->setBuffer( buf );
}

optix::Buffer OptiXMeshImpl::getGeoMaterialBuffer( optix::Geometry geo )
{
  return geo["material_buffer"]->getBuffer();
}

void OptiXMeshImpl::setGeoMaterialBuffer( optix::Geometry geo, optix::Buffer buf )
{
  geo["material_buffer"]->setBuffer( buf );
}
