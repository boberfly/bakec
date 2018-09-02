/*
Copyright 2018 Oscar Sebio Cajaraville

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "meshmapping.h"
#include "bvh.h"
#include "computeshaders.h"
#include "logging.h"
#include "mesh.h"
#include <cassert>

static const size_t k_groupSize = 64;
static const size_t k_workPerFrame = 1024 * 128;

namespace
{
	struct UniformsData
	{
		uint32_t workOffset;
		uint32_t coordsSize;
		uint32_t bvhSize;
		float _pad;
	};

	std::vector<Pix_GPUData> computePixels(const CompressedMapUV *map)
	{
		const size_t count = map->positions.size();
		std::vector<Pix_GPUData> pixels(count);
		for (size_t i = 0; i < count; ++i)
		{
			auto &pix = pixels[i];
			pix.p = map->positions[i];
			pix.d = map->directions[i];
		}
		return pixels;
	}

	std::vector<PixT_GPUData> computePixelsT(const CompressedMapUV *map)
	{
		const size_t count = map->positions.size();
		std::vector<PixT_GPUData> pixels(count);
		for (size_t i = 0; i < count; ++i)
		{
			auto &pix = pixels[i];
			pix.n = map->normals[i];
			pix.t = map->tangents[i];
			pix.b = map->bitangents[i];
		}
		return pixels;
	}

	void fillMeshData(
		const Mesh *mesh,
		const BVH& bvh,
		std::vector<BVHGPUData> &bvhs,
		std::vector<Vector4> &positions,
		std::vector<Vector4> &normals)
	{
		if (bvh.children.empty() &&
			bvh.triangles.empty())
		{
			// Children node without triangles? Skip it
			return;
		}

#if 1
		if (!bvh.children.empty())
		{
			// If one of the children does not contain any triangles
			// we can skip this node completely as it is an extry AABB test
			// for nothing
			if (bvh.children[0].subtreeTriangleCount > 0 && bvh.children[1].subtreeTriangleCount == 0)
			{
				fillMeshData(mesh, bvh.children[0], bvhs, positions, normals);
				return;
			}
			if (bvh.children[1].subtreeTriangleCount > 0 && bvh.children[0].subtreeTriangleCount == 0)
			{
				fillMeshData(mesh, bvh.children[1], bvhs, positions, normals);
				return;
			}
		}
#endif

		bvhs.emplace_back(BVHGPUData());
		BVHGPUData &d = bvhs.back();
		d.aabbMin = bvh.aabb.center - bvh.aabb.size;
		d.aabbMax = bvh.aabb.center + bvh.aabb.size;
		d.start = (uint32_t)positions.size();
		for (uint32_t tidx : bvh.triangles)
		{
			const auto &tri = mesh->triangles[tidx];
			const auto &v0 = mesh->vertices[tri.vertexIndex0];
			const auto &v1 = mesh->vertices[tri.vertexIndex1];
			const auto &v2 = mesh->vertices[tri.vertexIndex2];
			const auto p0 = mesh->positions[v0.positionIndex];
			const auto p1 = mesh->positions[v1.positionIndex];
			const auto p2 = mesh->positions[v2.positionIndex];
			positions.push_back(p0);
			positions.push_back(p1);
			positions.push_back(p2);
			normals.push_back(mesh->normals[v0.normalIndex]);
			normals.push_back(mesh->normals[v1.normalIndex]);
			normals.push_back(mesh->normals[v2.normalIndex]);
		}
		d.end = (uint32_t)positions.size();

		const size_t index = bvhs.size() - 1; // Because d gets invalidated by fillMeshData!
		if (bvh.children.size() > 0)
		{
			fillMeshData(mesh, bvh.children[0], bvhs, positions, normals);
			fillMeshData(mesh, bvh.children[1], bvhs, positions, normals);
		}
		bvhs[index].jump = (uint32_t)bvhs.size();
	}
}

void MeshMapping::init
(
	std::shared_ptr<const CompressedMapUV> map,
	std::shared_ptr<const Mesh> mesh,
	std::shared_ptr<const BVH> rootBVH,
	bool cullBackfaces
)
{
	// Pixels data
	{
		auto pixels = computePixels(map.get());
		_pixels = VBHandle(
			bgfx::createVertexBuffer(bgfx::copy(&pixels[0], sizeof(Pix_GPUData) * pixels.size()), computeDecl(sizeof(Pix_GPUData)), BGFX_BUFFER_COMPUTE_READ)
			, pixels.size());

		// Compute tangent data
		if (map->tangents.size() > 0)
		{
			auto pixelst = computePixelsT(map.get());
			_pixelst = VBHandle(
				bgfx::createVertexBuffer(bgfx::copy(&pixelst[0], sizeof(PixT_GPUData) * pixelst.size()), computeDecl(sizeof(PixT_GPUData)), BGFX_BUFFER_COMPUTE_READ)
				, pixelst.size());
		}
	}

	// Mesh data
	{
		std::vector<BVHGPUData> bvhs;
		std::vector<Vector4> positions;
		std::vector<Vector4> normals;
		fillMeshData(mesh.get(), *rootBVH, bvhs, positions, normals);
		_meshPositions = VBHandle(
			bgfx::createVertexBuffer(bgfx::copy(&positions[0], sizeof(Vector4) * positions.size()), computeDecl(sizeof(Vector4)), BGFX_BUFFER_COMPUTE_READ)
			, positions.size());
		_meshNormals = VBHandle(
			bgfx::createVertexBuffer(bgfx::copy(&normals[0], sizeof(Vector4) * normals.size()), computeDecl(sizeof(Vector4)), BGFX_BUFFER_COMPUTE_READ)
			, normals.size());
		_bvh = VBHandle(
			bgfx::createVertexBuffer(bgfx::copy(&bvhs[0], sizeof(BVHGPUData) * bvhs.size()), computeDecl(sizeof(BVHGPUData)), BGFX_BUFFER_COMPUTE_READ)
			, bvhs.size());
	}

	_workCount = ((map->positions.size() + k_groupSize - 1) / k_groupSize) * k_groupSize;

	// Results data
	{
		_coords = VBHandle(
			bgfx::createVertexBuffer(bgfx::alloc(_workCount), computeDecl(sizeof(Vector4)), BGFX_BUFFER_COMPUTE_WRITE)
			, _workCount);
		_tidx = VBHandle(
			bgfx::createVertexBuffer(bgfx::alloc(_workCount), computeDecl(sizeof(uint32_t)), BGFX_BUFFER_COMPUTE_WRITE)
			, _workCount);
	}

	// Shader
	{
		_program = ProgramHandle(LoadComputeShader_MeshMapping());
		_programCullBackfaces = ProgramHandle(LoadComputeShader_MeshMappingCullBackfaces());
	}

	// Uniforms
	{
		_uniforms = UniformHandle(
			bgfx::createUniform("u_params", bgfx::UniformType::Vec4, 1));
	}

	_cullBackfaces = cullBackfaces;

	_workOffset = 0;
}

bool MeshMapping::runStep()
{
	assert(_workOffset < _workCount);
	const size_t workLeft = _workCount - _workOffset;
	const size_t work = workLeft < k_workPerFrame ? workLeft : k_workPerFrame;
	assert(work % k_groupSize == 0);

	if (_workOffset == 0) _timing.begin();

	bgfx::ProgramHandle program;
	if (_cullBackfaces) program = _programCullBackfaces.handle;
	else program = _program.handle;

	UniformsData uniformsData;
	uniformsData.workOffset = _workOffset;
	uniformsData.coordsSize = _coords.size;
	uniformsData.bvhSize = _bvh.size;

	bgfx::setUniform(_uniforms.handle, &uniformsData, 1);
	bgfx::setBuffer(4, _pixels.handle, bgfx::Access::Read);
	bgfx::setBuffer(5, _meshPositions.handle, bgfx::Access::Read);
	bgfx::setBuffer(6, _bvh.handle, bgfx::Access::Read);
	bgfx::setBuffer(7, _coords.handle, bgfx::Access::Write);
	bgfx::setBuffer(8, _tidx.handle, bgfx::Access::Write);

	bgfx::dispatch(0, program, work / k_groupSize, 1, 1);

	_workOffset += work;

	if (_workOffset == _workCount)
	{
		_timing.end();
		logDebug("MeshMap", "Mesh mapping took " + std::to_string(_timing.elapsedSeconds()) + " seconds.");
	}

	return _workOffset >= _workCount;
}

MeshMappingTask::MeshMappingTask(std::shared_ptr<MeshMapping> meshmapping)
	: _meshMapping(meshmapping)
{
}

MeshMappingTask::~MeshMappingTask()
{
}

bool MeshMappingTask::runStep()
{
	assert(_meshMapping);
	return _meshMapping->runStep();
}

void MeshMappingTask::finish()
{
	//glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	uint32_t frame = bgfx::frame();
}

float MeshMappingTask::progress() const
{
	assert(_meshMapping);
	return _meshMapping->progress();
}
