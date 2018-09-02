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

bgfx::ProgramHandle LoadComputeShader_MeshMapping();
bgfx::ProgramHandle LoadComputeShader_MeshMappingCullBackfaces();

bgfx::ProgramHandle LoadComputeShader_AO_GenData();
bgfx::ProgramHandle LoadComputeShader_AO_Sampling();
bgfx::ProgramHandle LoadComputeShader_AO_Aggregate();

bgfx::ProgramHandle LoadComputeShader_BN_GenData();
bgfx::ProgramHandle LoadComputeShader_BN_Sampling();
bgfx::ProgramHandle LoadComputeShader_BN_Aggregate();

bgfx::ProgramHandle LoadComputeShader_Thick_GenData();
bgfx::ProgramHandle LoadComputeShader_Thick_Sampling();
bgfx::ProgramHandle LoadComputeShader_Thick_Aggregate();

bgfx::ProgramHandle LoadComputeShader_Height();
bgfx::ProgramHandle LoadComputeShader_Position();
bgfx::ProgramHandle LoadComputeShader_Normal();

bgfx::ProgramHandle LoadComputeShader_ToTangentSpace();