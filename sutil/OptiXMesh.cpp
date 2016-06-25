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


#include "OptiXMesh.h"
#include "OptiXMeshClasses.h"
#include "OptiXMeshImpl.h"

#include "AccelDescriptor.h"

#include <optixpp_namespace.h>

using namespace OptiXMeshClasses;


OptiXMesh::OptiXMesh( optix::Context context,
                      optix::GeometryGroup geometrygroup,
                      const AccelDescriptor& accel_desc )
: m_impl( new OptiXMeshImpl( context,
                             geometrygroup,
                             accel_desc ) )
{ }

OptiXMesh::OptiXMesh( optix::Context context,
                      optix::GeometryGroup geometrygroup,
                      optix::Material default_material,
                      const AccelDescriptor& accel_desc )
: m_impl( new OptiXMeshImpl( context,
                             geometrygroup,
                             default_material,
                             accel_desc ) )
{ }

OptiXMesh::~OptiXMesh() { if( m_impl ) delete m_impl; }


void OptiXMesh::loadBegin_Geometry( const std::string& filename ) { m_impl->beginLoad(filename); }
void OptiXMesh::loadFinish_Materials() { m_impl->finalizeLoad(); }

void OptiXMesh::setDefaultIntersectionProgram( optix::Program inx_program ) {
  m_impl->setDefaultIntersectionProgram( inx_program );
}
optix::Program OptiXMesh::getDefaultIntersectionProgram() const {
  return m_impl->getDefaultIntersectionProgram();
}

void OptiXMesh::setDefaultBoundingBoxProgram( optix::Program bbox_program ) {
  m_impl->setDefaultBoundingBoxProgram( bbox_program );
}
optix::Program OptiXMesh::getDefaultBoundingBoxProgram() const {
  return m_impl->getDefaultBoundingBoxProgram();
}

void OptiXMesh::setDefaultOptiXMaterial( optix::Material material ) {
  m_impl->setDefaultOptiXMaterial( material );
}
optix::Material OptiXMesh::getDefaultOptiXMaterial() const {
  return m_impl->getDefaultOptiXMaterial();
}

void OptiXMesh::setDefaultMaterialParams( const MeshMaterialParams& params ) {
  m_impl->setDefaultMaterialParams( params );
}
const MeshMaterialParams& OptiXMesh::getDefaultMaterialParams() const {
  return m_impl->getDefaultMaterialParams();
}

int OptiXMesh::getNumVertices() const { return m_impl->getNumVertices(); }
int OptiXMesh::getNumNormals() const { return m_impl->getNumNormals(); }
int OptiXMesh::getNumColors() const { return m_impl->getNumColors(); }
int OptiXMesh::getNumTextureCoordinates() const { return m_impl->getNumTextureCoordinates(); }

int OptiXMesh::getNumTriangles() const { return m_impl->getNumTriangles(); }

float* OptiXMesh::getVertexData() { return m_impl->getVertexData(); }
const float* OptiXMesh::getVertexData() const { return m_impl->getVertexData(); }

float* OptiXMesh::getNormalData() { return m_impl->getNormalData(); }
const float* OptiXMesh::getNormalData() const { return m_impl->getNormalData(); }

unsigned char* OptiXMesh::getColorData() { return m_impl->getColorData(); }
const unsigned char* OptiXMesh::getColorData() const { return m_impl->getColorData(); }

float* OptiXMesh::getTextureCoordinateData() { return m_impl->getTextureCoordinateData(); }
const float* OptiXMesh::getTextureCoordinateData() const { return m_impl->getTextureCoordinateData(); }

int OptiXMesh::getVertexStride() const { return m_impl->getVertexStride(); }
int OptiXMesh::getNormalStride() const { return m_impl->getNormalStride(); }
int OptiXMesh::getColorStride() const { return m_impl->getColorStride(); }
int OptiXMesh::getTextureCoordinateStride() const { return m_impl->getTextureCoordinateStride(); }

const float* OptiXMesh::getBBoxMin() const { return m_impl->getBBoxMin(); }
const float* OptiXMesh::getBBoxMax() const { return m_impl->getBBoxMax(); }

void OptiXMesh::updateBBox() { m_impl->updateBBox(); }

const std::string& OptiXMesh::getMaterialLibraryName() const {
  return m_impl->getMaterialLibraryName();
}

MeshGroup& OptiXMesh::getMeshGroup(const std::string& group_name) {
  return m_impl->getMeshGroup(group_name);
}

const MeshGroup& OptiXMesh::getMeshGroup(const std::string& group_name) const {
  return m_impl->getMeshGroup(group_name);
}

GroupGeometryInfo& OptiXMesh::getGroupGeometryInfo( const std::string& group_name ) {
  return m_impl->getGroupGeometryInfo(group_name);
}

const GroupGeometryInfo& OptiXMesh::getGroupGeometryInfo( const std::string& group_name ) const {
  return m_impl->getGroupGeometryInfo(group_name);
}

size_t OptiXMesh::getMaterialCount() const { return m_impl->getMaterialCount(); }

void OptiXMesh::setMeshMaterialParams( int i, const MeshMaterialParams& params ) {
  m_impl->setMeshMaterialParams( i, params );
}

MeshMaterialParams& OptiXMesh::getMeshMaterialParams( int i ) {
  return m_impl->getMeshMaterialParams( i );
}

const MeshMaterialParams& OptiXMesh::getMeshMaterialParams( int i ) const {
  return m_impl->getMeshMaterialParams( i );
}

void OptiXMesh::setOptiXMaterial( int i, optix::Material material ) {
  m_impl->setOptiXMaterial( i, material );
}

const optix::Material& OptiXMesh::getOptiXMaterial( int i ) const {
  return m_impl->getOptiXMaterial( i );
}


optix::Aabb OptiXMesh::getSceneBBox() const { return m_impl->getSceneBBox(); }
optix::Buffer OptiXMesh::getLightBuffer() const { return m_impl->getLightBuffer(); }

const optix::Matrix4x4& OptiXMesh::getLoadingTransform() const {
  return m_impl->getLoadingTransform();
}
  
void OptiXMesh::setLoadingTransform( const optix::Matrix4x4& transform ) {
  m_impl->setLoadingTransform( transform );
}


void OptiXMesh::addMaterial() {
  m_impl->addMaterial();
}

int OptiXMesh::getGroupMaterialNumber(const MeshGroup& group) const {
  return m_impl->getGroupMaterialNumber( group );
}


void OptiXMesh::setOptixInstanceMatParams( optix::GeometryInstance gi,
                                           const MeshMaterialParams& params ) const
{
  m_impl->setOptixInstanceMatParams( gi, params );
}

void OptiXMesh::setMergeMeshGroups( bool merge ) const 
{
  m_impl->setMergeMeshGroups( merge );
}
