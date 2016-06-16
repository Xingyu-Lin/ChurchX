
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

//-------------------------------------------------------------------------------
//
//  ppm.cpp -- Progressive photon mapping scene
//
//-------------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <iostream>
#include <GLUTDisplay.h>
#include <sutil.h>
#include <ImageLoader.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <limits>
#include "ppm.h"
#include "select.h"
#include "PpmObjLoader.h"
#include "random.h"
#include "model.h"
#include <fstream>
#include "ParticipatingMedium.h"
#include <vector>
#include "optixu_matrix_namespace.h"

using namespace optix;

#define NUM_VOLUMETRIC_PHOTONS 2000000

// For demo
static const bool sideWall = true;
static const bool golden = false;

static const bool frontLightSkew = false;

// Finds the smallest power of 2 greater or equal to x.
inline unsigned int pow2roundup(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x + 1;
}

inline float max(float a, float b)
{
	return a > b ? a : b;
}

inline RT_HOSTDEVICE int max_component(float3 a)
{
	if (a.x > a.y) {
		if (a.x > a.z) {
			return 0;
		}
		else {
			return 2;
		}
	}
	else {
		if (a.y > a.z) {
			return 1;
		}
		else {
			return 2;
		}
	}
}

static void print(float3 outVec)
{
	printf("(%f, %f, %f)\n", outVec.x, outVec.y, outVec.z);
}

float3 sphericalToCartesian(float theta, float phi)
{
	float cos_theta = cosf(theta);
	float sin_theta = sinf(theta);
	float cos_phi = cosf(phi);
	float sin_phi = sinf(phi);
	float3 v;
	v.x = cos_phi * sin_theta;
	v.z = sin_phi * sin_theta;
	v.y = cos_theta;
	return v;
}

enum SplitChoice {
	RoundRobin,
	HighestVariance,
	LongestDim
};

//-----------------------------------------------------------------------------
//
// Whitted Scene
//
//-----------------------------------------------------------------------------

class ProgressivePhotonScene : public SampleScene
{
public:
	ProgressivePhotonScene(unsigned int photon_launch_dim) : SampleScene()
			, m_frame_number(0)
			, m_display_debug_buffer(false)
			, m_print_timings(false)
			, m_cornell_box(false)
			, m_light_phi(-2.20f)
			, m_light_theta(1.15f)
			, m_photon_launch_width(photon_launch_dim)
			, m_photon_launch_height(photon_launch_dim)
			, m_split_choice(LongestDim)
	{
		m_num_photons = (m_photon_launch_width * m_photon_launch_height * ProgressivePhotonScene::MAX_PHOTON_COUNT);
	}

	// From SampleScene
	void   initScene(InitialCameraData& camera_data);
	bool   keyPressed(unsigned char key, int x, int y);
	void   trace(const RayGenCameraData& camera_data);
	void   doResize(unsigned int width, unsigned int height);
	void createLightParameters(const std::vector<float3> squareCor, float3& v1, float3& v2, float3& anchor);
	void createLights();

	Buffer getOutputBuffer();

	void setSceneCornellBox() { m_cornell_box = true; }
	void setSceneOBJ()        { m_cornell_box = false; }
	void printTimings()       { m_print_timings = true; }
	void displayDebugBuffer() { m_display_debug_buffer = true; }
	int loadObjConfig(const std::string &filename);
private:
	void createPhotonMap();
	void loadObjGeometry();
	void createCornellBoxGeometry();
	GeometryInstance createParallelogram(const float3& anchor,
										 const float3& offset1,
										 const float3& offset2,
										 const float3& color);

	enum ProgramEnum {
		rtpass,
		ppass,
		gather,
		clear_radiance_photon,
		clear_hitRecord,
		numPrograms
	};
	std::vector<std::string> m_church_parts_name;

	unsigned int  m_frame_number;
	bool          m_display_debug_buffer;
	bool          m_print_timings;
	bool          m_cornell_box;
	Program       m_pgram_bounding_box;
	Program       m_pgram_intersection;
	Material      m_material;
	Material      m_glass_material;

	Buffer        m_display_buffer;
	Buffer        m_photons;
	Buffer        m_photon_map;
	Buffer        m_debug_buffer;

	Buffer m_volumetricPhotonsBuffer;
	GeometryGroup m_volumetricPhotonsRoot;

	float         m_light_phi;
	float         m_light_theta;
	unsigned int  m_iteration_count;
	unsigned int  m_photon_map_size;
	unsigned int  m_photon_launch_width;
	unsigned int  m_photon_launch_height;
	unsigned int  m_num_photons;
	SplitChoice   m_split_choice;
	PPMLight      m_light;
	PPMLight*	  m_multiLights;
	int			  m_numLights;
	const static unsigned int WIDTH;
	const static unsigned int HEIGHT;
	const static unsigned int MAX_PHOTON_COUNT;
	const static float PPMRadius;

	const static float m_sigma_a;
	const static float m_sigma_s;
};

const unsigned int ProgressivePhotonScene::WIDTH = 768u;
const unsigned int ProgressivePhotonScene::HEIGHT = 768u;
const unsigned int ProgressivePhotonScene::MAX_PHOTON_COUNT = 5u;
const float ProgressivePhotonScene::PPMRadius = 2.0f;
const float ProgressivePhotonScene::m_sigma_a = 0.000f;
const float ProgressivePhotonScene::m_sigma_s = 0.001f;

bool ProgressivePhotonScene::keyPressed(unsigned char key, int x, int y)
{
	float step_size = 0.01f;
	bool light_changed = false;;
	switch (key)
	{
		case 'l':
			m_light_phi += step_size;
			if (m_light_phi >  M_PIf * 2.0f) m_light_phi -= M_PIf * 2.0f;
			light_changed = true;
			break;
		case 'j':
			m_light_phi -= step_size;
			if (m_light_phi <  0.0f) m_light_phi += M_PIf * 2.0f;
			light_changed = true;
			break;
		case 'k':
			std::cerr << "new theta: " << m_light_theta + step_size << " max: " << M_PIf / 2.0f << std::endl;
			m_light_theta = fminf(m_light_theta + step_size, M_PIf / 2.0f);
			light_changed = true;
			break;
		case 'i':
			std::cerr << "new theta: " << m_light_theta - step_size << " min: 0.0f " << std::endl;
			m_light_theta = fmaxf(m_light_theta - step_size, 0.0f);
			light_changed = true;
			break;
	}

	if (light_changed && !m_cornell_box) {
		std::cerr << " theta: " << m_light_theta << "  phi: " << m_light_phi << std::endl;
		m_light.position = 1000.0 * sphericalToCartesian(m_light_theta, m_light_phi);
		printf("%f, %f, %f", m_light.position.x, m_light.position.y, m_light.position.z);
		m_light.direction = normalize(make_float3(0.0f, 0.0f, 0.0f) - m_light.position);
		m_context["light"]->setUserData(sizeof(PPMLight), &m_light);
		signalCameraChanged();
		return true;
	}

	return false;
}

void ProgressivePhotonScene::initScene(InitialCameraData& camera_data)
{
	rtContextSetPrintEnabled(m_context->get(), true);
	// There's a performance advantage to using a device that isn't being used as a display.
	// We'll take a guess and pick the second GPU if the second one has the same compute
	// capability as the first.
	int deviceId = 0;
	int computeCaps[2];
	if (RTresult code = rtDeviceGetAttribute(0, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCaps))
		throw Exception::makeException(code, 0);
	for (unsigned int index = 1; index < Context::getDeviceCount(); ++index) {
		int computeCapsB[2];
		if (RTresult code = rtDeviceGetAttribute(index, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCapsB))
			throw Exception::makeException(code, 0);
		if (computeCaps[0] == computeCapsB[0] && computeCaps[1] == computeCapsB[1]) {
			deviceId = index;
			break;
		}
	}
	m_context->setDevices(&deviceId, &deviceId + 1);

	m_context->setRayTypeCount(num_ray_type);
	m_context->setEntryPointCount(numPrograms);
	m_context->setStackSize(4096);

	m_context["max_depth"]->setUint(3);
	m_context["max_photon_count"]->setUint(MAX_PHOTON_COUNT);
	m_context["scene_epsilon"]->setFloat(1.e-4f);
	m_context["alpha"]->setFloat(0.7f);
	m_context["total_emitted"]->setFloat(0.0f);
	m_context["frame_number"]->setFloat(0.0f);
	m_context["use_debug_buffer"]->setUint(m_display_debug_buffer ? 1 : 0);

	// Display buffer
	m_display_buffer = createOutputBuffer(RT_FORMAT_FLOAT4, WIDTH, HEIGHT);
	m_context["output_buffer"]->set(m_display_buffer);

	// Debug output buffer
	m_debug_buffer = m_context->createBuffer(RT_BUFFER_OUTPUT);
	m_debug_buffer->setFormat(RT_FORMAT_FLOAT4);
	m_debug_buffer->setSize(WIDTH, HEIGHT);
	m_context["debug_buffer"]->set(m_debug_buffer);

	// RTPass output buffer
	Buffer output_buffer = m_context->createBuffer(RT_BUFFER_OUTPUT);
	output_buffer->setFormat(RT_FORMAT_USER);
	output_buffer->setElementSize(sizeof(HitRecord));
	output_buffer->setSize(WIDTH, HEIGHT);
	m_context["rtpass_output_buffer"]->set(output_buffer);

	// RTPass pixel sample buffers
	Buffer image_rnd_seeds = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_INT2, WIDTH, HEIGHT);
	m_context["image_rnd_seeds"]->set(image_rnd_seeds);
	uint2* seeds = reinterpret_cast<uint2*>(image_rnd_seeds->map());
	for (unsigned int i = 0; i < WIDTH*HEIGHT; ++i)
		seeds[i] = random2u();
	image_rnd_seeds->unmap();

	// RTPass ray gen program
	{
		std::string ptx_path = ptxpath("progressivePhotonMap", "ppm_rtpass.cu");
		Program ray_gen_program = m_context->createProgramFromPTXFile(ptx_path, "rtpass_camera");
		m_context->setRayGenerationProgram(rtpass, ray_gen_program);

		// RTPass exception/miss programs
		Program exception_program = m_context->createProgramFromPTXFile(ptx_path, "rtpass_exception");
		m_context->setExceptionProgram(rtpass, exception_program);
		m_context["rtpass_bad_color"]->setFloat(1.0f, 1.0f, 1.0f);
		m_context->setMissProgram(rtpass, m_context->createProgramFromPTXFile(ptx_path, "rtpass_miss"));
		m_context["rtpass_bg_color"]->setFloat(make_float3(0.34f, 0.55f, 0.85f));
	}

	// clear hit record
	{
		std::string ptx_path = ptxpath("progressivePhotonMap", "HitRecordInitialize.cu");
		Program program = m_context->createProgramFromPTXFile(ptx_path, "kernel");
		m_context->setRayGenerationProgram(clear_hitRecord, program);
	}

	// Declare these so validation will pass
	m_context["rtpass_eye"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
	m_context["rtpass_U"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
	m_context["rtpass_V"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
	m_context["rtpass_W"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));

	// Photon pass
	m_photons = m_context->createBuffer(RT_BUFFER_OUTPUT);
	m_photons->setFormat(RT_FORMAT_USER);
	m_photons->setElementSize(sizeof(PhotonRecord));
	m_photons->setSize(m_num_photons);
	m_context["ppass_output_buffer"]->set(m_photons);


	{
		std::string ptx_path = ptxpath("progressivePhotonMap", "ppm_ppass.cu");
		Program ray_gen_program = m_context->createProgramFromPTXFile(ptx_path, "ppass_camera");
		m_context->setRayGenerationProgram(ppass, ray_gen_program);

		Buffer photon_rnd_seeds = m_context->createBuffer(RT_BUFFER_INPUT,
														  RT_FORMAT_UNSIGNED_INT2,
														  m_photon_launch_width,
														  m_photon_launch_height);
		uint2* seeds = reinterpret_cast<uint2*>(photon_rnd_seeds->map());
		for (unsigned int i = 0; i < m_photon_launch_width*m_photon_launch_height; ++i)
			seeds[i] = random2u();
		photon_rnd_seeds->unmap();
		m_context["photon_rnd_seeds"]->set(photon_rnd_seeds);

	}

	// Gather phase
	{
		std::string ptx_path = ptxpath("progressivePhotonMap", "ppm_gather.cu");
		Program gather_program = m_context->createProgramFromPTXFile(ptx_path, "gather");
		m_context->setRayGenerationProgram(gather, gather_program);
		Program exception_program = m_context->createProgramFromPTXFile(ptx_path, "gather_exception");
		m_context->setExceptionProgram(gather, exception_program);

		m_photon_map_size = pow2roundup(m_num_photons) - 1;
		m_photon_map = m_context->createBuffer(RT_BUFFER_INPUT);
		m_photon_map->setFormat(RT_FORMAT_USER);
		m_photon_map->setElementSize(sizeof(PhotonRecord));
		m_photon_map->setSize(m_photon_map_size);
		m_context["photon_map"]->set(m_photon_map);
	}

	// Populate scene hierarchy
	if (!m_cornell_box) {
		// Related to photonSphere
		{
			m_volumetricPhotonsBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
			m_volumetricPhotonsBuffer->setFormat(RT_FORMAT_USER);
			m_volumetricPhotonsBuffer->setElementSize(sizeof(Photon));
			m_volumetricPhotonsBuffer->setSize(NUM_VOLUMETRIC_PHOTONS);
			m_context["volumetricPhotons"]->setBuffer(m_volumetricPhotonsBuffer);

			optix::Geometry photonSpheres = m_context->createGeometry();
			photonSpheres->setPrimitiveCount(NUM_VOLUMETRIC_PHOTONS);
			std::string ptx_path = ptxpath("progressivePhotonMap", "VolumetricPhotonSphere.cu");
			photonSpheres->setIntersectionProgram(m_context->createProgramFromPTXFile(ptx_path, "intersect"));
			photonSpheres->setBoundingBoxProgram(m_context->createProgramFromPTXFile(ptx_path, "boundingBox"));

			optix::Material material = m_context->createMaterial();
			ptx_path = ptxpath("progressivePhotonMap", "VolumetricPhotonSphereRadiance.cu");
			material->setAnyHitProgram(volumetric_radiance, m_context->createProgramFromPTXFile(ptx_path, "anyHitRadiance"));
			optix::GeometryInstance volumetricPhotonSpheres = m_context->createGeometryInstance(photonSpheres, &material, &material + 1);
			volumetricPhotonSpheres["photonsBuffer"]->setBuffer(m_volumetricPhotonsBuffer);

			m_volumetricPhotonsRoot = m_context->createGeometryGroup();
			m_volumetricPhotonsRoot->setChildCount(1);
			optix::Acceleration m_volumetricPhotonSpheresAcceleration = m_context->createAcceleration("MedianBvh", "Bvh");
			m_volumetricPhotonsRoot->setAcceleration(m_volumetricPhotonSpheresAcceleration);
			m_volumetricPhotonsRoot->setChild(0, volumetricPhotonSpheres);
			m_context["volumetricPhotonsRoot"]->set(m_volumetricPhotonsRoot);
		}

		// Clear Volumetric Photons Program

		{
			std::string ptx_path = ptxpath("progressivePhotonMap", "VolumetricPhotonInitialize.cu");
			Program program = m_context->createProgramFromPTXFile(ptx_path, "kernel");
			m_context->setRayGenerationProgram(clear_radiance_photon, program);
		}

		optix::Aabb aabb;
		if (loadObjConfig("./data/churchdata/config.txt") == -1) return;
		loadObjGeometry();

		if (sideWall)
		camera_data = InitialCameraData(make_float3(-345.579, -141.678, -13.3228), // eye
										make_float3(236.489, -437.131, -414.981),      // lookat
										make_float3(0, 1, 0),     // up
										45.0f);                              // vfov
		else
		camera_data = InitialCameraData(make_float3(214.823, -103.356, 73.8156), // eye
										make_float3(-514.297, -213.848, -134.995),      // lookat
										make_float3(0, 1, 0),     // up
										55.0f);
		bool useWindowLight = true;
		if (!useWindowLight)
		{
			m_light.is_area_light = 0;
			m_light.position = 1000.0f * sphericalToCartesian(m_light_theta, m_light_phi);
			//printf("%f, %f, %f", m_light.position.x, m_light.position.y, m_light.position.z);
			//m_light.position = make_float3(-400.0f, -160.0f, -345.0f);
			m_light.direction = normalize(make_float3(0.0f, 0.0f, 0.0f) - m_light.position);
			m_light.radius = 250.0f * 0.01745329252f;
			m_light.power = make_float3(5.5e5f, 5.5e5f, 5.5e5f);
			m_context["light"]->setUserData(sizeof(PPMLight), &m_light);
		} else
		{
			createLights();
			m_context["light"]->setUserData(sizeof(PPMLight), m_multiLights );
			//TODO set to m_numLights * sizeof(PPMLight)
		}
		m_context["rtpass_default_radius2"]->setFloat(20.0f);// 0.25f);
		m_context["ambient_light"]->setFloat(0.1f, 0.1f, 0.1f);
		std::string full_path = std::string(sutilSamplesDir()) + "/tutorial/data/CedarCity.hdr";
		const float3 default_color = make_float3(0.8f, 0.88f, 0.97f);
		m_context["envmap"]->setTextureSampler(loadTexture(m_context, full_path, default_color));
	}
	else {

		createCornellBoxGeometry();
		// Set up camera
		camera_data = InitialCameraData(make_float3(278.0f, 273.0f, -850.0f), // eye
										make_float3(278.0f, 273.0f, 0.0f),    // lookat
										make_float3(0.0f, 1.0f, 0.0f),       // up
										35.0f);                                // vfov

		m_light.is_area_light = 1;
		m_light.anchor = make_float3(343.0f, 548.6f, 227.0f);
		m_light.v1 = make_float3(0.0f, 0.0f, 105.0f);
		m_light.v2 = make_float3(-130.0f, 0.0f, 0.0f);
		m_light.direction = normalize(cross(m_light.v1, m_light.v2));
		m_light.power = make_float3(0.5e6f, 0.4e6f, 0.2e6f);
		m_context["light"]->setUserData(sizeof(PPMLight), &m_light);
		m_context["rtpass_default_radius2"]->setFloat(400.0f);
		m_context["ambient_light"]->setFloat(0.0f, 0.0f, 0.0f);
		const float3 default_color = make_float3(0.0f, 0.0f, 0.0f);
		m_context["envmap"]->setTextureSampler(loadTexture(m_context, "", default_color));
	}


	// Prepare to run
	m_context->validate();
	m_context->compile();
}

Buffer ProgressivePhotonScene::getOutputBuffer()
{
	return m_display_buffer;
}

inline uchar4 makeColor(const float3& c)
{
	uchar4 pixel;
	pixel.x = static_cast<unsigned char>(fmaxf(fminf(c.z, 1.0f), 0.0f) * 255.99f);
	pixel.y = static_cast<unsigned char>(fmaxf(fminf(c.y, 1.0f), 0.0f) * 255.99f);
	pixel.z = static_cast<unsigned char>(fmaxf(fminf(c.x, 1.0f), 0.0f) * 255.99f);
	pixel.w = 0;
	return pixel;
}


bool photonCmpX(PhotonRecord* r1, PhotonRecord* r2) { return r1->position.x < r2->position.x; }
bool photonCmpY(PhotonRecord* r1, PhotonRecord* r2) { return r1->position.y < r2->position.y; }
bool photonCmpZ(PhotonRecord* r1, PhotonRecord* r2) { return r1->position.z < r2->position.z; }


void buildKDTree(PhotonRecord** photons, int start, int end, int depth, PhotonRecord* kd_tree, int current_root,
				 SplitChoice split_choice, float3 bbmin, float3 bbmax)
{
	// If we have zero photons, this is a NULL node
	if (end - start == 0) {
		kd_tree[current_root].axis = PPM_NULL;
		kd_tree[current_root].energy = make_float3(0.0f);
		return;
	}

	// If we have a single photon
	if (end - start == 1) {
		photons[start]->axis = PPM_LEAF;
		kd_tree[current_root] = *(photons[start]);
		return;
	}

	// Choose axis to split on
	int axis;
	switch (split_choice) {
		case RoundRobin:
		{
			axis = depth % 3;
		}
			break;
		case HighestVariance:
		{
			float3 mean = make_float3(0.0f);
			float3 diff2 = make_float3(0.0f);
			for (int i = start; i < end; ++i) {
				float3 x = photons[i]->position;
				float3 delta = x - mean;
				float3 n_inv = make_float3(1.0f / (static_cast<float>(i - start) + 1.0f));
				mean = mean + delta * n_inv;
				diff2 += delta*(x - mean);
			}
			float3 n_inv = make_float3(1.0f / (static_cast<float>(end - start) - 1.0f));
			float3 variance = diff2 * n_inv;
			axis = max_component(variance);
		}
			break;
		case LongestDim:
		{
			float3 diag = bbmax - bbmin;
			axis = max_component(diag);
		}
			break;
		default:
			axis = -1;
			std::cerr << "Unknown SplitChoice " << split_choice << " at " << __FILE__ << ":" << __LINE__ << "\n";
			exit(2);
			break;
	}

	int median = (start + end) / 2;
	PhotonRecord** start_addr = &(photons[start]);
#if 0
	switch (axis) {
	case 0:
		std::nth_element(start_addr, start_addr + median - start, start_addr + end - start, photonCmpX);
		photons[median]->axis = PPM_X;
		break;
	case 1:
		std::nth_element(start_addr, start_addr + median - start, start_addr + end - start, photonCmpY);
		photons[median]->axis = PPM_Y;
		break;
	case 2:
		std::nth_element(start_addr, start_addr + median - start, start_addr + end - start, photonCmpZ);
		photons[median]->axis = PPM_Z;
		break;
	}
#else
	switch (axis) {
		case 0:
			select<PhotonRecord*, 0>(start_addr, 0, end - start - 1, median - start);
			photons[median]->axis = PPM_X;
			break;
		case 1:
			select<PhotonRecord*, 1>(start_addr, 0, end - start - 1, median - start);
			photons[median]->axis = PPM_Y;
			break;
		case 2:
			select<PhotonRecord*, 2>(start_addr, 0, end - start - 1, median - start);
			photons[median]->axis = PPM_Z;
			break;
	}
#endif
	float3 rightMin = bbmin;
	float3 leftMax = bbmax;
	if (split_choice == LongestDim) {
		float3 midPoint = (*photons[median]).position;
		switch (axis) {
			case 0:
				rightMin.x = midPoint.x;
				leftMax.x = midPoint.x;
				break;
			case 1:
				rightMin.y = midPoint.y;
				leftMax.y = midPoint.y;
				break;
			case 2:
				rightMin.z = midPoint.z;
				leftMax.z = midPoint.z;
				break;
		}
	}

	kd_tree[current_root] = *(photons[median]);
	buildKDTree(photons, start, median, depth + 1, kd_tree, 2 * current_root + 1, split_choice, bbmin, leftMax);
	buildKDTree(photons, median + 1, end, depth + 1, kd_tree, 2 * current_root + 2, split_choice, rightMin, bbmax);
}


void ProgressivePhotonScene::createPhotonMap()
{
	PhotonRecord* photons_data = reinterpret_cast<PhotonRecord*>(m_photons->map());
	PhotonRecord* photon_map_data = reinterpret_cast<PhotonRecord*>(m_photon_map->map());

	for (unsigned int i = 0; i < m_photon_map_size; ++i) {
		photon_map_data[i].energy = make_float3(0.0f);
	}

	// Push all valid photons to front of list
	unsigned int valid_photons = 0;
	PhotonRecord** temp_photons = new PhotonRecord*[m_num_photons];
	for (unsigned int i = 0; i < m_num_photons; ++i) {
		if (fmaxf(photons_data[i].energy) > 0.0f) {
			temp_photons[valid_photons++] = &photons_data[i];
		}
	}
	if (m_display_debug_buffer) {
		std::cerr << " ** valid_photon/m_num_photons =  "
		<< valid_photons << "/" << m_num_photons
		<< " (" << valid_photons / static_cast<float>(m_num_photons) << ")\n";
	}

	// Make sure we aren't at most 1 less than power of 2
	valid_photons = valid_photons >= m_photon_map_size ? m_photon_map_size : valid_photons;

	float3 bbmin = make_float3(0.0f);
	float3 bbmax = make_float3(0.0f);
	if (m_split_choice == LongestDim) {
		bbmin = make_float3(std::numeric_limits<float>::max());
		bbmax = make_float3(-std::numeric_limits<float>::max());
		// Compute the bounds of the photons
		for (unsigned int i = 0; i < valid_photons; ++i) {
			float3 position = (*temp_photons[i]).position;
			bbmin = fminf(bbmin, position);
			bbmax = fmaxf(bbmax, position);
		}
	}

	// Now build KD tree
	buildKDTree(temp_photons, 0, valid_photons, 0, photon_map_data, 0, m_split_choice, bbmin, bbmax);

	delete[] temp_photons;
	m_photon_map->unmap();
	m_photons->unmap();
}

void ProgressivePhotonScene::trace(const RayGenCameraData& camera_data)
{
	Buffer output_buffer = m_context["rtpass_output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	output_buffer->getSize(buffer_width, buffer_height);

	m_frame_number = m_camera_changed ? 0u : m_frame_number + 1;

	m_context["frame_number"]->setFloat(static_cast<float>(m_frame_number));
	if (m_camera_changed)
	{
		m_camera_changed = false;
		m_context["rtpass_eye"]->setFloat(camera_data.eye);
		m_context["rtpass_U"]->setFloat(camera_data.U);
		m_context["rtpass_V"]->setFloat(camera_data.V);
		m_context["rtpass_W"]->setFloat(camera_data.W);
		m_context->launch(clear_hitRecord,
						  static_cast<unsigned int>(buffer_width),
						  static_cast<unsigned int>(buffer_height));
		m_iteration_count = 1;
		m_context["total_emitted"]->setFloat(0.0f);
	}
	if (0)
	{
		Photon *debug_buffer = reinterpret_cast<Photon *>(m_volumetricPhotonsBuffer->map());
		for (int i = 0; i < 100; ++i)
			std::cout << i << " " << debug_buffer[i].position.x << " " << debug_buffer[i].position.y << " " <<
			debug_buffer[i].position.z << std::endl;
		m_volumetricPhotonsBuffer->unmap();
	}
	// Clear volume photons
	m_context["volumetricRadius"]->setFloat(PPMRadius);
	{
		//nvtx::ScopedRange r( "OptixEntryPoint::PPM_CLEAR_VOLUMETRIC_PHOTONS_PASS" );
		m_context->launch(clear_radiance_photon, NUM_VOLUMETRIC_PHOTONS);
	}

	if (0)
	{
		Photon *debug_buffer = reinterpret_cast<Photon *>(m_volumetricPhotonsBuffer->map());
		for (int i = 0; i < 100; ++i)
			std::cout << i << " " << debug_buffer[i].position.x << " " << debug_buffer[i].position.y << " " <<
			debug_buffer[i].position.z << std::endl;
		m_volumetricPhotonsBuffer->unmap();
	}
	// Trace photons
	if (m_print_timings) std::cerr << "Starting photon pass   ... ";
	Buffer photon_rnd_seeds = m_context["photon_rnd_seeds"]->getBuffer();
	uint2* seeds = reinterpret_cast<uint2*>(photon_rnd_seeds->map());
	for (unsigned int i = 0; i < m_photon_launch_width*m_photon_launch_height; ++i)
		seeds[i] = random2u();
	photon_rnd_seeds->unmap();
	double t0, t1;
	sutilCurrentTime(&t0);
	m_context->launch(ppass,
					  static_cast<unsigned int>(m_photon_launch_width),
					  static_cast<unsigned int>(m_photon_launch_height));

	m_volumetricPhotonsRoot->getAcceleration()->markDirty();
	// By computing the total number of photons as an unsigned long long we avoid 32 bit
	// floating point addition errors when the number of photons gets sufficiently large
	// (the error of adding two floating point numbers when the mantissa bits no longer
	// overlap).
	// TODO:
	m_context["total_emitted"]->setFloat(static_cast<float>((unsigned long long)m_iteration_count*m_photon_launch_width*m_photon_launch_height));
	sutilCurrentTime(&t1);
	if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;

	// Build KD tree 
	if (m_print_timings) std::cerr << "Starting kd_tree build ... ";
	sutilCurrentTime(&t0);
	createPhotonMap();
	sutilCurrentTime(&t1);
	if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;

	// Trace viewing rays
	{
		if (m_print_timings) std::cerr << "Starting RT pass ... ";
		std::cerr.flush();
		double t0, t1;
		sutilCurrentTime(&t0);
		m_context->launch(rtpass,
						  static_cast<unsigned int>(buffer_width),
						  static_cast<unsigned int>(buffer_height));
		sutilCurrentTime(&t1);
		if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;
	}

	// Shade view rays by gathering photons
	if (m_print_timings) std::cerr << "Starting gather pass   ... ";
	sutilCurrentTime(&t0);
	m_context->launch(gather,
					  static_cast<unsigned int>(buffer_width),
					  static_cast<unsigned int>(buffer_height));
	sutilCurrentTime(&t1);
	if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;

	// Debug output
	if (m_display_debug_buffer) {
		sutilCurrentTime(&t0);
		float4* debug_data = reinterpret_cast<float4*>(m_debug_buffer->map());
		Buffer hit_records = m_context["rtpass_output_buffer"]->getBuffer();
		HitRecord* hit_record_data = reinterpret_cast<HitRecord*>(hit_records->map());
		float4 avg = make_float4(0.0f);
		float4 minv = make_float4(std::numeric_limits<float>::max());
		float4 maxv = make_float4(0.0f);
		float counter = 0.0f;
		for (unsigned int j = 0; j < buffer_height; ++j) {
			for (unsigned int i = 0; i < buffer_width; ++i) {
				/*
				if( i < 10 && j < 10 && 0) {
				fprintf( stderr, " %08.4f %08.4f %08.4f %08.4f\n", debug_data[j*buffer_width+i].x,
				debug_data[j*buffer_width+i].y,
				debug_data[j*buffer_width+i].z,
				debug_data[j*buffer_width+i].w );
				}
				*/


				if (hit_record_data[j*buffer_width + i].flags & PPM_HIT) {
					float4 val = debug_data[j*buffer_width + i];
					avg += val;
					minv = fminf(minv, val);
					maxv = fmaxf(maxv, val);
					counter += 1.0f;
				}
			}
		}
		m_debug_buffer->unmap();
		hit_records->unmap();

		avg = avg / counter;
		sutilCurrentTime(&t1);
		if (m_print_timings) std::cerr << "Stat collection time ...           " << t1 - t0 << std::endl;
		std::cerr << "(min, max, average):"
		<< " loop iterations: ( "
		<< minv.x << ", "
		<< maxv.x << ", "
		<< avg.x << " )"
		<< " radius: ( "
		<< minv.y << ", "
		<< maxv.y << ", "
		<< avg.y << " )"
		<< " N: ( "
		<< minv.z << ", "
		<< maxv.z << ", "
		<< avg.z << " )"
		<< " M: ( "
		<< minv.w << ", "
		<< maxv.w << ", "
		<< avg.w << " )";
		std::cerr << ", total_iterations = " << m_iteration_count;
		std::cerr << std::endl;
	}
	m_iteration_count++;
}


void ProgressivePhotonScene::doResize(unsigned int width, unsigned int height)
{
	// display buffer resizing handled in base class
	m_context["rtpass_output_buffer"]->getBuffer()->setSize(width, height);
	m_context["output_buffer"]->getBuffer()->setSize(width, height);
	m_context["image_rnd_seeds"]->getBuffer()->setSize(width, height);
	m_context["debug_buffer"]->getBuffer()->setSize(width, height);

	Buffer image_rnd_seeds = m_context["image_rnd_seeds"]->getBuffer();
	uint2* seeds = reinterpret_cast<uint2*>(image_rnd_seeds->map());
	for (unsigned int i = 0; i < width*height; ++i)
		seeds[i] = random2u();
	image_rnd_seeds->unmap();
}

GeometryInstance ProgressivePhotonScene::createParallelogram(const float3& anchor,
															 const float3& offset1,
															 const float3& offset2,
															 const float3& color)
{
	Geometry parallelogram = m_context->createGeometry();
	parallelogram->setPrimitiveCount(1u);
	parallelogram->setIntersectionProgram(m_pgram_intersection);
	parallelogram->setBoundingBoxProgram(m_pgram_bounding_box);

	float3 normal = normalize(cross(offset1, offset2));
	float d = dot(normal, anchor);
	float4 plane = make_float4(normal, d);

	float3 v1 = offset1 / dot(offset1, offset1);
	float3 v2 = offset2 / dot(offset2, offset2);

	parallelogram["plane"]->setFloat(plane);
	parallelogram["anchor"]->setFloat(anchor);
	parallelogram["v1"]->setFloat(v1);
	parallelogram["v2"]->setFloat(v2);

	GeometryInstance gi = m_context->createGeometryInstance(parallelogram,
															&m_material,
															&m_material + 1);
	gi["Kd"]->setFloat(color);
	gi["Ks"]->setFloat(0.0f, 0.0f, 0.0f);
	gi["use_grid"]->setUint(0u);
	gi["grid_color"]->setFloat(make_float3(0.0f));
	gi["emitted"]->setFloat(0.0f, 0.0f, 0.0f);
	return gi;
}

int ProgressivePhotonScene::loadObjConfig(const std::string &filename)
{
	std::fstream input;
	input.open(filename);
	if (!input)
	{
		std::cout << "Failed to open config file " << filename << std::endl;
		return -1;
	};
	std::string part_name;
	m_church_parts_name.clear();
	while (!input.eof())
	{
		input >> part_name;
		m_church_parts_name.push_back(part_name);
		input.get();
	}
	return 0;
}

void ProgressivePhotonScene::loadObjGeometry()
{
	//createMaterial
	std::string path1 = std::string(sutilSamplesPtxDir()) + "/progressivePhotonMap_generated_ppm_rtpass.cu.ptx";
	std::string path2 = std::string(sutilSamplesPtxDir()) + "/progressivePhotonMap_generated_ppm_ppass.cu.ptx";
	std::string path3 = std::string(sutilSamplesPtxDir()) + "/progressivePhotonMap_generated_ppm_gather.cu.ptx";

	Program closest_hit1 = m_context->createProgramFromPTXFile(path1, "rtpass_closest_hit");
	Program closest_hit2 = m_context->createProgramFromPTXFile(path2, "ppass_closest_hit");
	Program any_hit = m_context->createProgramFromPTXFile(path3, "gather_any_hit");
	m_material = m_context->createMaterial();
	m_material->setClosestHitProgram(rtpass_ray_type, closest_hit1);
	m_material->setClosestHitProgram(ppass_and_gather_ray_type, closest_hit2);
	m_material->setAnyHitProgram(shadow_ray_type, any_hit);

	m_material->setClosestHitProgram(radiance_in_participating_medium, closest_hit1);
	m_material->setClosestHitProgram(photon_in_participating_medium, closest_hit2);


	std::string path4 = std::string(sutilSamplesPtxDir()) + "/progressivePhotonMap_generated_glass.cu.ptx";
	Program closest_glass = m_context->createProgramFromPTXFile(path4, "ppass_closest_hit_transparent");
	m_glass_material = m_context->createMaterial();
	m_glass_material->setClosestHitProgram(rtpass_ray_type, closest_hit1);
	m_glass_material->setClosestHitProgram(ppass_and_gather_ray_type, closest_glass);
	m_glass_material->setAnyHitProgram(shadow_ray_type, any_hit);

	m_glass_material->setClosestHitProgram(radiance_in_participating_medium, closest_hit1);
	m_glass_material->setClosestHitProgram(photon_in_participating_medium, closest_glass);

	std::string path = std::string(sutilSamplesPtxDir()) + "/progressivePhotonMap_generated_triangle_mesh.cu.ptx";
	Program mesh_intersect = m_context->createProgramFromPTXFile(path, "mesh_intersect");
	Program mesh_bbox = m_context->createProgramFromPTXFile(path, "mesh_bounds");

	// Place all in group
	Group TopGroup = m_context->createGroup();
	std::vector<Model> church_parts(m_church_parts_name.size());

	TopGroup->setChildCount(m_church_parts_name.size());
	std::string relPath = std::string(sutilSamplesDir()) + "/progressivePhotonMap/data/churchdata/";


	for (int i = 0; i<m_church_parts_name.size(); ++i)
	{
		std::string objFullPath = relPath + m_church_parts_name[i] + ".obj";
		float3 translate = make_float3(0,50.4, 0);
		float3 scale = make_float3(20, 20, 20);
		float3 rotateAxis = make_float3(0, 0, 0);
		float rotateRadius = 0.0f;
		optix::Transform trans = m_context->createTransform();
		GeometryGroup tempGroup = m_context->createGeometryGroup();
		if (m_church_parts_name[i] == "staklo")
			church_parts[i] = Model(objFullPath, m_glass_material, m_accel_desc, NULL, mesh_intersect, mesh_bbox, m_context,
								i ? church_parts.back().m_geom_group : static_cast<GeometryGroup>(NULL), translate, scale, rotateRadius, rotateAxis);
		else
			church_parts[i] = Model(objFullPath, m_material, m_accel_desc, NULL, mesh_intersect, mesh_bbox, m_context,
			i ? church_parts.back().m_geom_group : static_cast<GeometryGroup>(NULL), translate, scale, rotateRadius, rotateAxis);
	}

	for (int i = 0; i<m_church_parts_name.size(); ++i)
		TopGroup->setChild(i, church_parts[i].m_tran);
	//GeometryInstance floorInstance  = m_context->createGeometryInstance(parallelogram, &m_material, &m_material + 1);
	//floorInstance["emitted"]->setFloat(0.0f, 0.0f, 0.0f);
	//floorInstance["Kd"]->setFloat(0.78, 0.78, 0.78);
	//floorInstance["Ks"]->setFloat(0.0, 0.0, 0.0);
	//floorInstance["diffuse_map"]->setTextureSampler(m_context->createTextureSampler());
	//GeometryGroup floorGroup = m_context->createGeometryGroup();
	//floorGroup->setChildCount(1);
	//floorGroup->setChild(0, floorInstance);
	//floorGroup->setAcceleration(m_context->createAcceleration("NoAccel", "NoAccel"));

	//TopGroup->setChildCount(TopGroup->getChildCount() + 1);
	//TopGroup->setChild(TopGroup->getChildCount() - 1, floorGroup);

	//Participating media
	{
		ParticipatingMedium partmedium = ParticipatingMedium(0.05, 0.01);
		optix::Aabb box;
		int boxsize = 500;
		box.set(make_float3(-boxsize, -boxsize, -boxsize), make_float3(boxsize, boxsize, boxsize));
		optix::Geometry geometry = m_context->createGeometry();

		//AABInstance participatingMediumCube (partmedium, box); ==
		geometry->setPrimitiveCount(1u);
		std::string ptx_path = ptxpath("progressivePhotonMap", "ABB.cu");
		geometry->setBoundingBoxProgram(m_context->createProgramFromPTXFile(ptx_path, "boundingBox"));
		geometry->setIntersectionProgram(m_context->createProgramFromPTXFile(ptx_path, "intersect"));
		float3 min = box.m_min;
		geometry["cuboidMin"]->setFloat(box.m_min);
		geometry["cuboidMax"]->setFloat(box.m_max);
		std::string partmedium_ptxpath = ptxpath("progressivePhotonMap", "ParticipatingMedium.cu");
		optix::Material ptM = partmedium.getOptixMaterial(m_context, partmedium_ptxpath);

		GeometryInstance gi = m_context->createGeometryInstance(geometry, &ptM, &ptM + 1);
		gi["sigma_a"]->setFloat(m_sigma_a);
		gi["sigma_s"]->setFloat(m_sigma_s);
		optix::GeometryGroup group = m_context->createGeometryGroup();
		group->setChildCount(1);
		group->setChild(0, gi);
		optix::Acceleration a = m_context->createAcceleration("Sbvh", "Bvh");
		group->setAcceleration(a);

		TopGroup->setChildCount(TopGroup->getChildCount() + 1);
		TopGroup->setChild(TopGroup->getChildCount() - 1, group);
	}

	TopGroup->setAcceleration(m_context->createAcceleration("Sbvh", "Bvh"));
	m_context["top_object"]->set(TopGroup);
	m_context["top_shadower"]->set(TopGroup);

}



void ProgressivePhotonScene::createCornellBoxGeometry()
{
	// Set up material
	m_material = m_context->createMaterial();
	m_material->setClosestHitProgram(rtpass, m_context->createProgramFromPTXFile(ptxpath("progressivePhotonMap", "ppm_rtpass.cu"),
																				 "rtpass_closest_hit"));
	m_material->setClosestHitProgram(ppass, m_context->createProgramFromPTXFile(ptxpath("progressivePhotonMap", "ppm_ppass.cu"),
																				"ppass_closest_hit"));
	m_material->setAnyHitProgram(gather, m_context->createProgramFromPTXFile(ptxpath("progressivePhotonMap", "ppm_gather.cu"),
																			 "gather_any_hit"));


	std::string ptx_path = ptxpath("progressivePhotonMap", "parallelogram.cu");
	m_pgram_bounding_box = m_context->createProgramFromPTXFile(ptx_path, "bounds");
	m_pgram_intersection = m_context->createProgramFromPTXFile(ptx_path, "intersect");


	// create geometry instances
	std::vector<GeometryInstance> gis;

	const float3 white = make_float3(0.8f, 0.8f, 0.8f);
	const float3 green = make_float3(0.05f, 0.8f, 0.05f);
	const float3 red = make_float3(0.8f, 0.05f, 0.05f);
	const float3 black = make_float3(0.0f, 0.0f, 0.0f);
	const float3 light = make_float3(15.0f, 15.0f, 5.0f);

	// Floor
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
									  make_float3(0.0f, 0.0f, 559.2f),
									  make_float3(556.0f, 0.0f, 0.0f),
									  white));

	// Ceiling
	gis.push_back(createParallelogram(make_float3(0.0f, 548.8f, 0.0f),
									  make_float3(556.0f, 0.0f, 0.0f),
									  make_float3(0.0f, 0.0f, 559.2f),
									  white));

	// Back wall
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 559.2f),
									  make_float3(0.0f, 548.8f, 0.0f),
									  make_float3(556.0f, 0.0f, 0.0f),
									  white));

	// Right wall
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
									  make_float3(0.0f, 548.8f, 0.0f),
									  make_float3(0.0f, 0.0f, 559.2f),
									  green));

	// Left wall
	gis.push_back(createParallelogram(make_float3(556.0f, 0.0f, 0.0f),
									  make_float3(0.0f, 0.0f, 559.2f),
									  make_float3(0.0f, 548.8f, 0.0f),
									  red));

	// Short block
	gis.push_back(createParallelogram(make_float3(130.0f, 165.0f, 65.0f),
									  make_float3(-48.0f, 0.0f, 160.0f),
									  make_float3(160.0f, 0.0f, 49.0f),
									  white));
	gis.push_back(createParallelogram(make_float3(290.0f, 0.0f, 114.0f),
									  make_float3(0.0f, 165.0f, 0.0f),
									  make_float3(-50.0f, 0.0f, 158.0f),
									  white));
	gis.push_back(createParallelogram(make_float3(130.0f, 0.0f, 65.0f),
									  make_float3(0.0f, 165.0f, 0.0f),
									  make_float3(160.0f, 0.0f, 49.0f),
									  white));
	gis.push_back(createParallelogram(make_float3(82.0f, 0.0f, 225.0f),
									  make_float3(0.0f, 165.0f, 0.0f),
									  make_float3(48.0f, 0.0f, -160.0f),
									  white));
	gis.push_back(createParallelogram(make_float3(240.0f, 0.0f, 272.0f),
									  make_float3(0.0f, 165.0f, 0.0f),
									  make_float3(-158.0f, 0.0f, -47.0f),
									  white));

	// Tall block
	gis.push_back(createParallelogram(make_float3(423.0f, 330.0f, 247.0f),
									  make_float3(-158.0f, 0.0f, 49.0f),
									  make_float3(49.0f, 0.0f, 159.0f),
									  white));
	gis.push_back(createParallelogram(make_float3(423.0f, 0.0f, 247.0f),
									  make_float3(0.0f, 330.0f, 0.0f),
									  make_float3(49.0f, 0.0f, 159.0f),
									  white));
	gis.push_back(createParallelogram(make_float3(472.0f, 0.0f, 406.0f),
									  make_float3(0.0f, 330.0f, 0.0f),
									  make_float3(-158.0f, 0.0f, 50.0f),
									  white));
	gis.push_back(createParallelogram(make_float3(314.0f, 0.0f, 456.0f),
									  make_float3(0.0f, 330.0f, 0.0f),
									  make_float3(-49.0f, 0.0f, -160.0f),
									  white));
	gis.push_back(createParallelogram(make_float3(265.0f, 0.0f, 296.0f),
									  make_float3(0.0f, 330.0f, 0.0f),
									  make_float3(158.0f, 0.0f, -49.0f),
									  white));

	// Light
	gis.push_back(createParallelogram(make_float3(343.0f, 548.7f, 227.0f),
									  make_float3(0.0f, 0.0f, 105.0f),
									  make_float3(-130.0f, 0.0f, 0.0f),
									  black));
	gis.back()["emitted"]->setFloat(light);


	// Create geometry group
	GeometryGroup geometry_group = m_context->createGeometryGroup();
	geometry_group->setChildCount(static_cast<unsigned int>(gis.size()));
	for (unsigned int i = 0; i < gis.size(); ++i)
		geometry_group->setChild(i, gis[i]);
	geometry_group->setAcceleration(m_context->createAcceleration("Bvh", "Bvh"));

	m_context["top_object"]->set(geometry_group);
	m_context["top_shadower"]->set(geometry_group);
}


//-----------------------------------------------------------------------------
//
// Main driver
//
//-----------------------------------------------------------------------------


void printUsageAndExit(const std::string& argv0, bool doExit = true)
{
	std::cerr
	<< "Usage  : " << argv0 << " [options]\n"
	<< "App options:\n"
	<< "  -h  | --help                               Print this usage message\n"
	<< "  -t  | --timeout <sec>                      Seconds before stopping rendering. Set to 0 for no stopping.\n"
	<< "        --cornell-box                        Display Cornell Box scene\n"
	<< "        --photon-dim <photons>               Width and height of photon launch grid. Default = 512.\n"
	#ifndef RELEASE_PUBLIC
	<< "  -pt | --print-timings                      Print timing information\n"
	<< " -ddb | --display-debug-buffer               Display the debug buffer information\n"
	#endif
	<< std::endl;
	GLUTDisplay::printUsage();

	std::cerr
	<< "App keystrokes:\n"
	<< "  i Move light up\n"
	<< "  j Move light left\n"
	<< "  k Move light down\n"
	<< "  l Move light right\n"
	<< std::endl;

	if (doExit) exit(1);
}

void ProgressivePhotonScene::createLightParameters(const std::vector<float3> squareCor, float3& v1, float3& v2, float3& anchor)
{
	//sqareCor contains v0 -----> v1
	//                   |
	//					 V
	//					 v2
	float3 translate = optix::make_float3(0, 50.4, 0);
	float3 scale = optix::make_float3(20, 20, 20);
	float3 rotateAxis = optix::make_float3(0, 0, 0);
	float rotateRadius = 0.0f;

	//transform matrix
	Matrix4x4 XForm = Matrix4x4::identity();
	XForm = Matrix4x4::scale(scale) * XForm;
	XForm = Matrix4x4::rotate(rotateRadius, rotateAxis) * XForm;
	XForm = Matrix4x4::translate(translate) * XForm;

	std::vector <float3> transCor;
	for (int i = 0; i < 3; ++i) {
		float4 newCor = XForm * make_float4(squareCor[i], 1.0);
		newCor /= newCor.w;
		transCor.push_back(make_float3(newCor.x, newCor.y, newCor.z));
	}
	v1 = transCor[1] - transCor[0];
	v2 = transCor[2] - transCor[0];
	anchor = transCor[0];
	return;
}

void ProgressivePhotonScene::createLights() {
	m_numLights = 1;
	m_multiLights = new PPMLight[m_numLights];

	std::vector<std::vector<float3> > squareCors;
	std::vector<float3> tmpSquareCors;

	//front side wall light
	if (!sideWall)
	{
		if (frontLightSkew) {
			tmpSquareCors.clear();
			tmpSquareCors.push_back(optix::make_float3(-35.3, -9.0, 6.3));
			tmpSquareCors.push_back(optix::make_float3(-39.3, -9.0, -6.3));
			tmpSquareCors.push_back(optix::make_float3(-20.2, 8.1, 6.2));
			squareCors.push_back(tmpSquareCors);
		}
		else {
			tmpSquareCors.clear();
			tmpSquareCors.push_back(optix::make_float3(-35.3, -9.0, 6.3));
			tmpSquareCors.push_back(optix::make_float3(-35.3, -9.0, -6.3));
			tmpSquareCors.push_back(optix::make_float3(-20.2, 8.1, 6.2));
			squareCors.push_back(tmpSquareCors);
		}
	}
	/*tmpSquareCors.clear();
	tmpSquareCors.push_back(optix::make_float3(-24.3, -4.1, 6.3));
	tmpSquareCors.push_back(optix::make_float3(-26.3, -4.1, -6.3));
	tmpSquareCors.push_back(optix::make_float3(-20.2, 0.0, 6.2));
	squareCors.push_back(tmpSquareCors);*/

	//left side wall light
	if (sideWall)
	{
		tmpSquareCors.clear();
		tmpSquareCors.push_back(optix::make_float3(-19.13, -14.00, -14.20));
		tmpSquareCors.push_back(optix::make_float3(-6.00, -14.00, -14.20));
		tmpSquareCors.push_back(optix::make_float3(-19.13, -8.76, -10.30));
		squareCors.push_back(tmpSquareCors);
	}
	
	float lightPowerScale;
	if (sideWall)
		lightPowerScale = 5.0;
	else
		lightPowerScale = 8.0;

	for (int i = 0; i < m_numLights; ++i) {
		if (!golden)
			m_multiLights[i].power = lightPowerScale * make_float3(0.4e6f, 0.5e6f, 0.5e6f);
		else
			m_multiLights[i].power = lightPowerScale * make_float3(0.6e6f, 0.4e6f, 0.2e6f);
		m_multiLights[i].is_area_light = 1;
		createLightParameters(squareCors[i], m_multiLights[i].v1, m_multiLights[i].v2, m_multiLights[i].anchor);
		m_multiLights[i].direction = normalize(cross(m_multiLights[i].v1, m_multiLights[i].v2));
		//Matrix4x4 Rot = Matrix4x4::rotate(0, m_multiLights[i].v1);
		//m_multiLights[i].v1 = make_float3(Rot * make_float4(m_multiLights[i].v1, 1.0f));
		//printf("v1: "); print(m_multiLights[i].v1);
		//printf("v2: "); print(m_multiLights[i].v2);
	}

}

int main(int argc, char** argv)
{

	GLUTDisplay::init(argc, argv);
	srand(time(0));
	bool print_timings = false;
	bool display_debug_buffer = false;
	bool cornell_box = false;
	float timeout = -1.0f;
	unsigned int photon_launch_dim = 512u;

	for (int i = 1; i < argc; ++i) {
		std::string arg(argv[i]);
		if (arg == "--help" || arg == "-h") {
			printUsageAndExit(argv[0]);
		}
		else if (arg == "--print-timings" || arg == "-pt") {
			print_timings = true;
		}
		else if (arg == "--display-debug-buffer" || arg == "-ddb") {
			display_debug_buffer = true;
		}
		else if (arg == "--cornell-box") {
			cornell_box = true;
		}
		else if (arg == "--photon-dim") {
			if (++i < argc) {
				photon_launch_dim = static_cast<unsigned int>(atoi(argv[i]));
			}
			else {
				std::cerr << "Missing argument to " << arg << "\n";
				printUsageAndExit(argv[0]);
			}
		}
		else if (arg == "--timeout" || arg == "-t") {
			if (++i < argc) {
				timeout = static_cast<float>(atof(argv[i]));
			}
			else {
				std::cerr << "Missing argument to " << arg << "\n";
				printUsageAndExit(argv[0]);
			}
		}
		else {
			std::cerr << "Unknown option: '" << arg << "'\n";
			printUsageAndExit(argv[0]);
		}
	}

	if (!GLUTDisplay::isBenchmark()) printUsageAndExit(argv[0], false);

	try {
		ProgressivePhotonScene scene(photon_launch_dim);
		if (print_timings) scene.printTimings();
		if (display_debug_buffer) scene.displayDebugBuffer();
		if (cornell_box) scene.setSceneCornellBox();
		GLUTDisplay::setProgressiveDrawingTimeout(timeout);
		GLUTDisplay::setUseSRGB(true);
		GLUTDisplay::run("ProgressivePhotonScene", &scene, GLUTDisplay::CDAnimated);
	}
	catch (Exception& e){
		sutilReportError(e.getErrorString().c_str());
		exit(1);
	}

	return 0;
}
