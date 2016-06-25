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

#include "HostMesh.h"
#include "HostMeshImpl.h"

HostMesh::HostMesh() : m_impl( new HostMeshImpl ) { }

HostMesh::~HostMesh() { if( m_impl ) delete m_impl; }

void HostMesh::loadModel( const std::string& filename ) { m_impl->loadModel(filename); }

int HostMesh::getNumVertices() const { return m_impl->getNumVertices(); }
int HostMesh::getNumNormals() const { return m_impl->getNumNormals(); }
int HostMesh::getNumColors() const { return m_impl->getNumColors(); }
int HostMesh::getNumTextureCoordinates() const { return m_impl->getNumTextureCoordinates(); }

int HostMesh::getNumTriangles() const { return m_impl->getNumTriangles(); }

float* HostMesh::getVertexData() { return m_impl->getVertexData(); }
const float* HostMesh::getVertexData() const { return m_impl->getVertexData(); }

float* HostMesh::getNormalData() { return m_impl->getNormalData(); }
const float* HostMesh::getNormalData() const { return m_impl->getNormalData(); }

unsigned char* HostMesh::getColorData() { return m_impl->getColorData(); }
const unsigned char* HostMesh::getColorData() const { return m_impl->getColorData(); }

float* HostMesh::getTextureCoordinateData() { return m_impl->getTextureCoordinateData(); }
const float* HostMesh::getTextureCoordinateData() const {
  return m_impl->getTextureCoordinateData();
}

int HostMesh::getVertexStride() const { return m_impl->getVertexStride(); }
int HostMesh::getNormalStride() const { return m_impl->getNormalStride(); }
int HostMesh::getColorStride() const { return m_impl->getColorStride(); }
int HostMesh::getTextureCoordinateStride() const {
  return m_impl->getTextureCoordinateStride();
}

const float* HostMesh::getBBoxMin() const { return m_impl->getBBoxMin(); }
const float* HostMesh::getBBoxMax() const { return m_impl->getBBoxMax(); }

void HostMesh::updateBBox() { m_impl->updateBBox(); }

const std::string& HostMesh::getMaterialLibraryName() const {
  return m_impl->getMaterialLibraryName();
}


MeshGroup& HostMesh::getMeshGroup(const std::string& group_name) {
  return m_impl->getMeshGroup(group_name);
}

const MeshGroup& HostMesh::getMeshGroup(const std::string& group_name) const {
  return m_impl->getMeshGroup(group_name);
}

size_t HostMesh::getMaterialCount() const { return m_impl->getMaterialCount(); }


MeshMaterialParams& HostMesh::getMeshMaterialParams( int i ) {
  return m_impl->getMeshMaterialParams(i);
}

const MeshMaterialParams& HostMesh::getMeshMaterialParams( int i ) const {
  return m_impl->getMeshMaterialParams(i);
}


int* HostMesh::getVertexIndices() { return m_impl->getVertexIndices(); }
const int* HostMesh::getVertexIndices() const { return m_impl->getVertexIndices(); }
int* HostMesh::getNormalIndices() { return m_impl->getNormalIndices(); }
const int* HostMesh::getNormalIndices() const { return m_impl->getNormalIndices(); }
int* HostMesh::getColorIndices() { return m_impl->getColorIndices(); }
const int* HostMesh::getColorIndices() const { return m_impl->getColorIndices(); }
int* HostMesh::getTextureCoordinateIndices() {
  return m_impl->getTextureCoordinateIndices();
}
const int* HostMesh::getTextureCoordinateIndices() const {
  return m_impl->getTextureCoordinateIndices();
}
