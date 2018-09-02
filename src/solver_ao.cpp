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

#include "solver_ao.h"
#include "compute.h"
#include "computeshaders.h"
#include "logging.h"
#include "meshmapping.h"
#include <cassert>

#include "image.h"

static const size_t k_groupSize = 64;
static const size_t k_workPerFrame = 1024 * 128;
static const size_t k_samplePermCount = 64 * 64;

namespace
{
	struct UniformsData
	{
		uint32_t workOffset;
		float _pad0;
		uint32_t bvhSize;
		float _pad1;
	};

	std::vector<Vector3> computeSamples(size_t sampleCount, size_t permutationCount)
	{
		const size_t count = sampleCount * permutationCount;
		std::vector<Vector3> sampleDirs(count);
		computeSamplesImportanceCosDir(sampleCount, permutationCount, &sampleDirs[0]);
		return sampleDirs;
	}
}

void AmbientOcclusionSolver::init(std::shared_ptr<const CompressedMapUV> map, std::shared_ptr<MeshMapping> meshMapping)
{
	_rayProgram = LoadComputeShader_AO_GenData();
	_aoProgram = LoadComputeShader_AO_Sampling();
	_avgProgram = LoadComputeShader_AO_Aggregate();
	_uvMap = map;
	_meshMapping = meshMapping;
	_workCount = ((map->positions.size() + k_groupSize - 1) / k_groupSize) * k_groupSize;

	{
		ShaderParams params;
		params.sampleCount = (uint32_t)_params.sampleCount;
		params.samplePermCount = (uint32_t)k_samplePermCount;
		params.minDistance = _params.minDistance;
		params.maxDistance = _params.maxDistance;
		_paramsCB = VBHandle(
			bgfx::createVertexBuffer(bgfx::copy(&params, sizeof(ShaderParams)), computeDecl(sizeof(ShaderParams)), BGFX_BUFFER_COMPUTE_READ_WRITE));
	}

	auto samples = computeSamples(_params.sampleCount, k_samplePermCount);
	std::vector<Vector4> samplesData(samples.begin(), samples.end());
	_samplesCB = VBHandle(
		bgfx::createVertexBuffer(bgfx::copy(&samplesData[0], sizeof(Vector4) * samplesData.size()), computeDecl(sizeof(Vector4)), BGFX_BUFFER_COMPUTE_READ_WRITE)
		, samplesData.size());

	uint32_t count = k_workPerFrame / _params.sampleCount;
	_rayDataCB = VBHandle(
		bgfx::createVertexBuffer(bgfx::alloc(sizeof(RayData) * count), computeDecl(sizeof(RayData)), BGFX_BUFFER_COMPUTE_WRITE)
		, count);

	_resultsMiddleCB = VBHandle(
		bgfx::createVertexBuffer(bgfx::alloc(sizeof(float) * k_workPerFrame), computeDecl(sizeof(float)), BGFX_BUFFER_COMPUTE_WRITE)
		, k_workPerFrame);

	//_resultsFinalCB = VBHandle(
	//	bgfx::createVertexBuffer(bgfx::alloc(sizeof(float) * _workCount), computeDecl(sizeof(float)), BGFX_BUFFER_COMPUTE_WRITE)
	//	, _workCount);
	_resultsFinalCB = TextureHandle(
		bgfx::createTexture2D(
		  _workCount
		, 1
		, false
		, 1
		, bgfx::TextureFormat::RGBA32F
		, BGFX_TEXTURE_NONE|BGFX_SAMPLER_NONE
		, bgfx::alloc(sizeof(float) * _workCount)
		));

	_workOffset = 0;
}

bool AmbientOcclusionSolver::runStep()
{
	const size_t totalWork = _workCount * _params.sampleCount;
	assert(_workOffset < totalWork);
	const size_t workLeft = totalWork - _workOffset;
	const size_t work = workLeft < k_workPerFrame ? workLeft : k_workPerFrame;
	assert(work % k_groupSize == 0);

	if (_workOffset == 0) _timing.begin();

	UniformsData uniformsData;
	uniformsData.workOffset = uint32_t(_workOffset / _params.sampleCount);
	uniformsData.bvhSize = _meshMapping->meshBVH().size;

	// Ray
	bgfx::setUniform(_uniforms.handle, &uniformsData, 1);

	bgfx::setBuffer(2, _meshMapping->meshPositions().handle, bgfx::Access::Read);
	bgfx::setBuffer(3, _meshMapping->meshNormals().handle, bgfx::Access::Read);
	bgfx::setBuffer(4, _meshMapping->coords().handle, bgfx::Access::Read);
	bgfx::setBuffer(5, _meshMapping->coords_tidx().handle, bgfx::Access::Read);
	bgfx::setBuffer(6, _rayDataCB.handle, bgfx::Access::Write);

	bgfx::dispatch(0, _rayProgram.handle, work / k_groupSize, 1, 1);

	// AO
	bgfx::setUniform(_uniforms.handle, &uniformsData, 1);

	bgfx::setBuffer(3, _paramsCB.handle, bgfx::Access::Read);
	bgfx::setBuffer(4, _meshMapping->meshPositions().handle, bgfx::Access::Read);
	bgfx::setBuffer(5, _meshMapping->meshBVH().handle, bgfx::Access::Read);
	bgfx::setBuffer(6, _samplesCB.handle, bgfx::Access::Read);
	bgfx::setBuffer(7, _rayDataCB.handle, bgfx::Access::Read);
	bgfx::setBuffer(8, _resultsMiddleCB.handle, bgfx::Access::Write);

	bgfx::dispatch(1, _aoProgram.handle, work / k_groupSize, 1, 1);

	// Avg
	bgfx::setUniform(_uniforms.handle, &uniformsData, 1);

	bgfx::setBuffer(2, _paramsCB.handle, bgfx::Access::Read);
	bgfx::setBuffer(3, _resultsMiddleCB.handle, bgfx::Access::Read);
	//bgfx::setBuffer(4, _resultsFinalCB.handle, bgfx::Access::Write);
	bgfx::setImage(4, _resultsFinalCB.handle, 0, bgfx::Access::Write, bgfx::TextureFormat::RGBA32F);

	bgfx::dispatch(2, _avgProgram.handle, work / k_groupSize, 1, 1);

	_workOffset += work;

	if (_workOffset >= totalWork)
	{
		_timing.end();
		logDebug("AO",
			"Ambient Occlusion map took " + std::to_string(_timing.elapsedSeconds()) +
			" seconds for " + std::to_string(_uvMap->width) + "x" + std::to_string(_uvMap->height));
	}

	return _workOffset >= totalWork;
}

float* AmbientOcclusionSolver::getResults()
{
	//assert(_sampleIndex >= _params.sampleCount);
	//glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	float* data = new float[_uvMap->width*_uvMap->height];
	uint32_t actualFrame = bgfx::frame();
	uint32_t expectedFrame = bgfx::readTexture(_resultsFinalCB.handle, &data, 0);
	if(actualFrame < expectedFrame)
	{
		bgfx::frame();
		expectedFrame = bgfx::readTexture(_resultsFinalCB.handle, &data, 0);
		if(actualFrame != expectedFrame)
		{
			delete[] data;
			return nullptr;
		}
	}
	return data;
}

AmbientOcclusionTask::AmbientOcclusionTask(std::unique_ptr<AmbientOcclusionSolver> solver, const char *outputPath, int dilation)
	: _solver(std::move(solver))
	, _outputPath(outputPath)
	, _dilation(dilation)
{
}

AmbientOcclusionTask::~AmbientOcclusionTask()
{
}

bool AmbientOcclusionTask::runStep()
{
	assert(_solver);
	return _solver->runStep();
}

void AmbientOcclusionTask::finish()
{
	assert(_solver);
	float *results = _solver->getResults();
	exportFloatImage(results, _solver->uvMap().get(), _outputPath.c_str(), true, _dilation); // TODO: Normalize
	delete[] results;
}

float AmbientOcclusionTask::progress() const
{
	return _solver->progress();
}