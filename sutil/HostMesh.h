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

#ifndef __samples_util_host_mesh_h__
#define __samples_util_host_mesh_h__

#include <string>
#include "sutilapi.h"
#include "HostMeshImpl.h"

class MeshGroup;
class MeshMaterialParams;

/**
 * This class's load methods allocate host-side arrays according to the number of
 * vertices, indices, and other data counted during the first pass of loading.
 * Objects of this class therefore own the loaded data.
 *
 * MeshBase only represents vertex indices, etc., as they are contained in groups,
 * but this class adds single object-level arrays to accommodate actually owning the
 * pool of vertex indices, normal indices, etc., where the pointers to index arrays
 * within each group don't own the arrays.  The loader then arranges the group index
 * array pointers to point into the proper locations within the pool arrays.
 *
 * For now, for expedience, this class only loads vertices and vertex indices,
 * but none of the other attributes, since all the samples that use it only need
 * vertices--not normals, etc.
 */
class SUTILCLASSAPI HostMesh
{
public:
  
  SUTILAPI HostMesh();
  // Deallocates all owned *_indices pointers
  SUTILAPI ~HostMesh();

  SUTILAPI void loadModel( const std::string& filename );

  /**
   * Calls functor( group ) for each group in the mesh, where 'group' is of
   * the type MeshGroup.
   */
  template <class Functor>
  void forEachGroup( Functor functor ) const;

  template <class Functor>
  void forEachGroup( Functor functor );

  SUTILAPI int getNumVertices() const;
  SUTILAPI int getNumNormals() const;
  SUTILAPI int getNumColors() const;
  SUTILAPI int getNumTextureCoordinates() const;

  SUTILAPI int getNumTriangles() const;

  SUTILAPI float* getVertexData();
  SUTILAPI const float* getVertexData() const;

  SUTILAPI float* getNormalData();
  SUTILAPI const float* getNormalData() const;
  
  SUTILAPI unsigned char* getColorData();
  SUTILAPI const unsigned char* getColorData() const;
  
  SUTILAPI float* getTextureCoordinateData();
  SUTILAPI const float* getTextureCoordinateData() const;

  SUTILAPI int getVertexStride() const;
  SUTILAPI int getNormalStride() const;
  SUTILAPI int getColorStride() const;
  SUTILAPI int getTextureCoordinateStride() const;

  SUTILAPI const float* getBBoxMin() const;
  SUTILAPI const float* getBBoxMax() const;

  SUTILAPI void updateBBox();

  SUTILAPI const std::string& getMaterialLibraryName() const;

  SUTILAPI MeshGroup& getMeshGroup(const std::string& group_name);
  SUTILAPI const MeshGroup& getMeshGroup(const std::string& group_name) const;

  SUTILAPI size_t getMaterialCount() const;
  SUTILAPI MeshMaterialParams& getMeshMaterialParams( int i );
  SUTILAPI const MeshMaterialParams& getMeshMaterialParams( int i ) const;

  SUTILAPI int* getVertexIndices();
  SUTILAPI const int* getVertexIndices() const;
  SUTILAPI int* getNormalIndices();
  SUTILAPI const int* getNormalIndices() const;
  SUTILAPI int* getColorIndices();
  SUTILAPI const int* getColorIndices() const;
  SUTILAPI int* getTextureCoordinateIndices();
  SUTILAPI const int* getTextureCoordinateIndices() const;

private:

  HostMeshImpl* m_impl;
};


template <class Functor>
void HostMesh::forEachGroup( Functor functor ) const
{
  m_impl->forEachGroup( functor );
}


template <class Functor>
void HostMesh::forEachGroup( Functor functor )
{
  m_impl->forEachGroup( functor );
}


#endif /* __samples_util_host_mesh_h__ */
