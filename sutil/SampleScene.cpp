
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

#if defined(__APPLE__)
#  include <OpenGL/gl.h>
#else
#  include <GL/glew.h>
#  if defined(_WIN32)
#    include <GL/wglew.h>
#  endif
#  include <GL/gl.h>
#endif

#include <SampleScene.h>

#include <optixu/optixu_math_stream_namespace.h>
#include <optixu/optixu.h>

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>


using namespace optix;

//-----------------------------------------------------------------------------
// 
// SampleScene class implementation 
//
//-----------------------------------------------------------------------------


const int SampleScene::IMAGE_WIDTH  = 1024;
const int SampleScene::IMAGE_HEIGHT = 768;


SampleScene::SampleScene()
  : m_camera_changed( true ), m_use_pbo_buffer( true ), m_num_devices( 0 ), m_cpu_rendering_enabled( false ),
  m_vca_enabled( false ), mImageWidth( IMAGE_WIDTH ), mImageHeight( IMAGE_HEIGHT )
{
  m_context = Context::create();
  updateCPUMode();
}

namespace {

const int VCA_RETRY_SLEEP_SECONDS = 10;
const int VCA_MAX_RETRY_COUNT = 6;

RemoteDevice create_remote_device( const VCAOptions& opts )
{
  if (opts.url.empty()) {
    std::cerr << "No VCA cluster manager URL specified. Running locally." << std::endl;
    return 0;
  }

  RemoteDevice rdev(0);
  try {
    rdev = RemoteDevice::create( opts.url.c_str(), opts.user.c_str(), opts.password.c_str() );
  }
  catch (Exception& e){
    sutilReportError( e.getErrorString().c_str() );
    std::cerr << "Remote VCA cluster manager login failed. Running locally." << std::endl;
  }
  if (!rdev) return 0;

  // Query compatible packages
  unsigned int num_configs;
  rdev->getAttribute( RT_REMOTEDEVICE_ATTRIBUTE_NUM_CONFIGURATIONS, sizeof(unsigned int), &num_configs );

  if (num_configs == 0) {
    std::cerr << "No compatible configurations found on this remote device." << std::endl;
    rdev->destroy();
    return 0;
  }

  // Display configurations
  std::cerr << "Available configurations:" << std::endl;
  for( unsigned int i=0; i<num_configs; ++i ) {
    std::string cfg_name = rdev->getConfiguration(i);
    std::cerr << " [" << i << "] " << cfg_name << std::endl;
  }

  try {
    std::cerr << "Reserving " << opts.num_nodes << " nodes with configuration " << opts.config_index << std::endl;
    rdev->reserve(opts.num_nodes, opts.config_index);
  }
  catch (Exception& e) {
    sutilReportError( e.getErrorString().c_str() );
    std::cerr << "Remote VCA cluster manager configuration failed. Running locally." << std::endl;
    rdev->destroy();
    return 0;
  }

  std::cerr << "Waiting for the cluster to be ready..." << std::endl;
  int retry_count = 0;
  while( retry_count++ < VCA_MAX_RETRY_COUNT )
  {
    int ready;
    rdev->getAttribute(  RT_REMOTEDEVICE_ATTRIBUTE_STATUS, sizeof(int), &ready );
    if( ready == RT_REMOTEDEVICE_STATUS_READY ) {
      break;
    }
    sutilSleep( VCA_RETRY_SLEEP_SECONDS );
  }
  if (retry_count <= VCA_MAX_RETRY_COUNT) {
    std::cerr << "...done." << std::endl;
  } else {
    std::cerr << "...timed out." << std::endl;
    rdev->release();
    rdev->destroy();
  }
  return rdev;  
}

}  // anon namespace

SampleScene::SampleScene( const VCAOptions& vca_options )
  : m_camera_changed( true ), m_use_pbo_buffer( false ), m_num_devices( 0 ), m_cpu_rendering_enabled( false ),
  m_vca_enabled( true ), mImageWidth( IMAGE_WIDTH ), mImageHeight( IMAGE_HEIGHT )
{
  m_context = Context::create();
  m_remote_device = create_remote_device(vca_options);
  if (m_remote_device) m_context->setRemoteDevice(m_remote_device);
}

SampleScene::InitialCameraData::InitialCameraData( const std::string &camstr)
{
  std::istringstream istr(camstr);
  istr >> eye >> lookat >> up >> vfov;
}


const char* const SampleScene::ptxpath( const std::string& target, const std::string& base )
{
  static std::string path;
  path = std::string(sutilSamplesPtxDir()) + "/" + target + "_generated_" + base + ".ptx";
  return path.c_str();
}

Buffer SampleScene::createOutputBuffer( RTformat format,
                                        unsigned int width,
                                        unsigned int height )
{
  // Set number of devices to be used
  // Default, 0, means not to specify them here, but let OptiX use its default behavior.
  if(m_num_devices)
  {
    int max_num_devices    = Context::getDeviceCount();
    int actual_num_devices = std::min( max_num_devices, std::max( 1, m_num_devices ) );
    std::vector<int> devs(actual_num_devices);
    for( int i = 0; i < actual_num_devices; ++i ) devs[i] = i;
    m_context->setDevices( devs.begin(), devs.end() );
  }

  Buffer buffer;

  if ( m_use_pbo_buffer )
  {
    assert(!m_vca_enabled && "pixel buffer objects are not compatible with VCA mode");

    // First allocate the memory for the GL buffer, then attach it to OptiX.
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    size_t element_size;
    m_context->checkError(rtuGetSizeForRTformat(format, &element_size));
    glBufferData(GL_ARRAY_BUFFER, element_size * width * height, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, vbo);
    buffer->setFormat(format);
    buffer->setSize( width, height );
  }
  else {
    buffer = m_context->createBuffer( RT_BUFFER_OUTPUT, format, width, height);
  }

  return buffer;
}

void SampleScene::cleanUp()
{
  m_context->destroy();
  m_context = 0;

  if (m_remote_device) {
    std::cerr << "Releasing remote VCA" << std::endl;
    m_remote_device->release();
    m_remote_device->destroy();
    m_remote_device = 0;
  }
}

void
SampleScene::trace( const RayGenCameraData& camera_data, bool& display )
{
  trace(camera_data);
}

void SampleScene::resize(unsigned int width, unsigned int height)
{
  if (width == mImageWidth && height == mImageHeight) return;

  try {
    Buffer buffer = getOutputBuffer();
    buffer->setSize( width, height );
    mImageWidth = width;
    mImageHeight = height;

    if(m_use_pbo_buffer)
    {
      buffer->unregisterGLBuffer();
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer->getGLBOId());
      glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer->getElementSize() * width * height, 0, GL_STREAM_DRAW);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      buffer->registerGLBuffer();
    }

  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(2);
  }

  // Let the user resize any other buffers
  doResize( width, height );
}

void
SampleScene::setNumDevices( int ndev )
{
  m_num_devices = ndev;

  if (m_cpu_rendering_enabled && m_num_devices > 0) {
    rtContextSetAttribute(m_context.get()->get(), RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS, sizeof(m_num_devices), &m_num_devices);
  }
}

void
SampleScene::enableCPURendering(bool enable)
{
  // Is CPU mode already enabled
  std::vector<int> devices = m_context->getEnabledDevices();
  bool isCPUEnabled = false;
  for(std::vector<int>::const_iterator iter = devices.begin(); iter != devices.end(); ++iter)
  {
    if (m_context->getDeviceName(*iter) == "CPU") {
      isCPUEnabled = true;
      break;
    }
  }

  // Already in desired state, good-bye.
  if (isCPUEnabled == enable)
    return;

  if (enable)
  {
    // Turn on CPU mode

    assert(!m_vca_enabled && "CPU mode is not compatible with VCA mode");

    int ordinal;
    for(ordinal = m_context->getDeviceCount()-1; ordinal >= 0; ordinal--)
    {
      if (m_context->getDeviceName(ordinal) == "CPU") {
        break;
      }
    }
    if (ordinal < 0)
      throw Exception("Attempting to enable CPU mode, but no CPU device found");
    m_context->setDevices(&ordinal, &ordinal+1);
  } else
  {
    // Turn off CPU mode

    // For now, simply grab the first device
    int ordinal = 0;
    m_context->setDevices(&ordinal, &ordinal+1);
  }

  // Check this here, in case we failed to make it into GPU mode.
  updateCPUMode();
}

void
SampleScene::incrementCPUThreads(int delta)
{
  int num_threads;
  RTresult code = rtContextGetAttribute(m_context.get()->get(), RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS, sizeof(num_threads), &num_threads);
  m_context->checkError(code);
  num_threads += delta;
  if (num_threads <= 0)
    num_threads = 1;
  setNumDevices(num_threads);
}

// Checks to see if CPU mode has been enabled and sets the appropriate flags.
void
SampleScene::updateCPUMode()
{
  m_cpu_rendering_enabled = m_context->getDeviceName(m_context->getEnabledDevices()[0]) == "CPU";
  if (m_cpu_rendering_enabled) {
    m_use_pbo_buffer = false;
    assert(!m_vca_enabled && "CPU mode is not compatible with VCA mode");
  }
}
