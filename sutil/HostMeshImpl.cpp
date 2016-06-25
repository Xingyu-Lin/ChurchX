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

#include "HostMeshImpl.h"

HostMeshImpl::HostMeshImpl() :
  Base(),
  m_vertex_indices( 0 ),
  m_normal_indices( 0 ),
  m_color_indices( 0 ),
  m_texture_coordinate_indices( 0 )
{ }


HostMeshImpl::~HostMeshImpl()
{
  if( getVertexData() ) delete[] getVertexData();
  if( getNormalData() ) delete[] getNormalData();
  if( getColorData() ) delete[] getColorData();
  if( getTextureCoordinateData() ) delete[] getTextureCoordinateData();
  if( m_vertex_indices ) delete[] m_vertex_indices;
  if( m_normal_indices ) delete[] m_normal_indices;
  if( m_color_indices ) delete[] m_color_indices;
  if( m_texture_coordinate_indices ) delete[] m_texture_coordinate_indices;
}

// No pre-processing needed
void HostMeshImpl::preProcess() { }


// ----------------------------------------------------------------------------
// allocateData() and its utility functor
//
struct HostMeshImpl::AllocateGroupsFunctor
{
  HostMeshImpl& m_mesh;
  int       m_triangles_index;

  AllocateGroupsFunctor( HostMeshImpl& mesh ) : 
    m_mesh( mesh ), m_triangles_index( 0 )
  { }

  // Assumes that m_mesh.vertex_indices, etc., have already been initialized
  void operator()( MeshGroup& group ) {
    group.vertex_indices = m_mesh.m_vertex_indices + m_triangles_index * 3;
    
    m_triangles_index += group.num_triangles;
  }
};

void HostMeshImpl::allocateData()
{
  setVertexData( new float[getNumVertices() * 3] );
  m_vertex_indices = new int[getNumTriangles() * 3];
  
  // Tell Base::loadData*() not to write other attributes
  setNormalData( 0 );
  setColorData( 0 );
  setTextureCoordinateData( 0 );

  forEachGroup( AllocateGroupsFunctor(*this) );

  // Vertices should be stored compactly
  setVertexStride( 0 );
}


// ----------------------------------------------------------------------------
// Other virtual overrides
//

// No special preparation needed; data pointers are already set by allocateData
void HostMeshImpl::startWritingData() { }

// No post-processing needed
void HostMeshImpl::postProcess() { }

void HostMeshImpl::finishWritingData() { }
