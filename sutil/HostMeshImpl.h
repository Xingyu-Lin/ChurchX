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

#ifndef __samples_util_host_mesh_impl_h__
#define __samples_util_host_mesh_impl_h__

#include "MeshBase.h"


class HostMeshImpl : public MeshBase
{
public:
  
  typedef MeshBase Base;

  HostMeshImpl();

  virtual ~HostMeshImpl();

  int* getVertexIndices() { return m_vertex_indices; }
  const int* getVertexIndices() const { return m_vertex_indices; }
  int* getNormalIndices() { return m_normal_indices; }
  const int* getNormalIndices() const { return m_normal_indices; }
  int* getColorIndices() { return m_color_indices; }
  const int* getColorIndices() const { return m_color_indices; }
  int* getTextureCoordinateIndices() { return m_texture_coordinate_indices; }
  const int* getTextureCoordinateIndices() const { return m_texture_coordinate_indices; }

protected:

  // Implementation of MeshBase's subclass interface
  virtual void preProcess();
  virtual void allocateData();
  virtual void startWritingData();
  virtual void postProcess();
  virtual void finishWritingData();

private:

  // A helper functor for allocateData()
  struct AllocateGroupsFunctor;
 
  int* m_vertex_indices;
  int* m_normal_indices;
  int* m_color_indices;
  int* m_texture_coordinate_indices;
};


#endif /* __samples_util_host_mesh_impl_h__ */
