#pragma once

#include "compute.h"
#include "fornos.h"
#include "math.h"
#include "timing.h"
#include <cstdint>
#include <memory>

struct CompressedMapUV;
class Mesh;
class BVH;

struct Pix_GPUData
{
	Vector3 p;
	float _pad0;
	Vector3 d;
	float _pad1;
};

struct PixT_GPUData
{
	Vector3 n;
	float _pad0;
	Vector3 t;
	float _pad1;
	Vector3 b;
	float _pad2;
};

struct BVHGPUData
{
	Vector3 aabbMin;
	Vector3 aabbMax;
	uint32_t start, end;
	uint32_t jump;
	BVHGPUData() : aabbMin(), aabbMax(), start(0), end(0), jump(0) {}
};

class MeshMapping
{
public:
	void init(std::shared_ptr<const CompressedMapUV> map, std::shared_ptr<const Mesh> mesh, std::shared_ptr<const BVH> rootBVH, bool cullBackfaces = false);
	bool runStep();

	inline float progress() const { return (float)_workOffset / (float)_workCount; }

	inline const VBHandle coords() const { return _coords; }
	inline const VBHandle coords_tidx() const { return _tidx; }
	inline const VBHandle pixels() const { return _pixels; }
	inline const VBHandle pixelst() const { return _pixelst; }
	inline const VBHandle meshPositions() const { return _meshPositions; }
	inline const VBHandle meshNormals() const { return _meshNormals; }
	inline const VBHandle meshBVH() const { return _bvh; }


private:
	size_t _workOffset;
	size_t _workCount;
	bool _cullBackfaces = false;

	VBHandle _coords;
	VBHandle _tidx;
	VBHandle _pixels;
	VBHandle _pixelst;
	VBHandle _meshPositions;
	VBHandle _meshNormals;
	VBHandle _bvh;
	ProgramHandle _program;
	ProgramHandle _programCullBackfaces;
	UniformHandle _uniforms;

	Timing _timing;
};

class MeshMappingTask : public FornosTask
{
public:
	MeshMappingTask(std::shared_ptr<MeshMapping> meshmapping);
	~MeshMappingTask();

	bool runStep();
	void finish();
	float progress() const;
	const char* name() const { return "Mesh mapping"; }

private:
	std::shared_ptr<MeshMapping> _meshMapping;
};