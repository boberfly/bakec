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

#pragma once

#include <bgfx/bgfx.h>
#include <string>
#include <cassert>
#include <vector>
#include "math.h"

class Mesh;

bgfx::ProgramHandle CreateComputeProgram(const char *path);

bgfx::ProgramHandle CreateComputeProgramFromMemory(const char *src);

bgfx::VertexDecl computeDecl(uint8_t stride)
{
	bgfx::VertexDecl vertDecl;
	vertDecl.begin().skip(stride).end();
	return vertDecl;
}

template <typename T>
class BgfxHandle
{
public:
	BgfxHandle(T handle, size_t size = 1) : handle(handle), size(size) {}
	~BgfxHandle() { bgfx::destroy(_handle); }
	T handle;
	size_t size;
};

typedef BgfxHandle<bgfx::VertexBufferHandle> VBHandle;
typedef BgfxHandle<bgfx::TextureHandle> TextureHandle;
typedef BgfxHandle<bgfx::UniformHandle> UniformHandle;
typedef BgfxHandle<bgfx::ProgramHandle> ProgramHandle;

/// Stores per-pixel data for the low-poly mesh
struct MapUV
{
	std::vector<Vector3> positions;
	std::vector<Vector3> directions;
	std::vector<Vector3> normals;
	std::vector<Vector3> tangents;
	std::vector<Vector3> bitangents;

	const uint32_t width;
	const uint32_t height;

	MapUV(uint32_t w, uint32_t h)
		: width(w)
		, height(h)
		, positions(w * h, Vector3())
		, directions(w * h, Vector3())
		, normals(w * h, Vector3())
	{
	}

	/// Builds a map from a mesh
	/// @param mesh Mesh
	/// @param width Map width
	/// @param height Map height
	static MapUV* fromMesh(const Mesh *mesh, uint32_t width, uint32_t height);
	static MapUV* fromMeshes(const Mesh *mesh, const Mesh *meshDirs, uint32_t width, uint32_t height);
	static MapUV* fromMeshes_Hybrid(const Mesh *mesh, const Mesh *meshDirs, uint32_t width, uint32_t height, float edge);
};

/// MapUV without any pixels with no data
/// This is for a more efficient processing in the GPU
struct CompressedMapUV
{
	std::vector<Vector3> positions;
	std::vector<Vector3> directions;
	std::vector<Vector3> normals;
	std::vector<Vector3> tangents;
	std::vector<Vector3> bitangents;
	std::vector<uint32_t> indices; // Actual index in the MapUV

	const uint32_t width;
	const uint32_t height;

	/// Creates a compressed map from a raw map
	CompressedMapUV(const MapUV *map);
};
