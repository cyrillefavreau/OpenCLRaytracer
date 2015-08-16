/* 
 * OpenCL Raytracer
 * Copyright (C) 2011-2012 Cyrille Favreau <cyrille_favreau@hotmail.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. 
 */

/*
 * Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
 *
 */

#pragma once

#include <CL/opencl.h>

#include "DLL_API.h"
#include <stdio.h>
#include <string>
#include <windows.h>
#if USE_KINECT
#include <nuiapi.h>
#endif // USE_KINECT

const int gTextureWidth  = 512;
const int gTextureHeight = 512;
const int gTextureDepth  = 3;
const int gColorDepth    = 4;

enum KernelSourceType
{
   kst_file,
   kst_string
};

enum PrimitiveType 
{
   ptSphere = 0,
   ptTriangle,
   ptCheckboard,
   ptCamera,
   ptXYPlane,
   ptYZPlane,
   ptXZPlane,
   ptCylinder
};

const int NO_MATERIAL = -1;

const int gKinectColorVideo = 4;
const int gVideoWidth       = 640;
const int gVideoHeight      = 480;
const int gKinectColorDepth = 2;
const int gDepthWidth       = 320;
const int gDepthHeight      = 240;

struct Material
{
   cl_float4 color;
   cl_float  refraction;
   cl_int    textured;
   cl_float  transparency;
   cl_int    textureId;
   cl_float4 specular;
};

struct Primitive
{
   cl_float4 center;
   //cl_float4 rotation;
   cl_float4 size;
   cl_int    type;
   cl_int    materialId;
   cl_float  materialRatioX;
   cl_float  materialRatioY;
};

struct Lamp
{
   cl_float4 center;
   cl_float4 color;
};

class OPENCLRAYTRACERMODULE_API OpenCLKernel
{
public:
   OpenCLKernel( int platformId, int device, int nbWorkingItems, int draft );
   ~OpenCLKernel();

public:
   // ---------- Devices ----------
   void initializeDevice(
      int        width, 
      int        height, 
      int        nbPrimitives,
      int        nbLamps,
      int        nbMaterials,
      int        nbTextures,
      BYTE*      bitmap);
   void releaseDevice();

   void compileKernels(
      const KernelSourceType sourceType,
      const std::string& source, 
      const std::string& ptxFileName,
      const std::string& options);

public:
   // ---------- Rendering ----------
   void render(
      int   imageW, 
      int   imageH, 
      BYTE* bitmap,
      float time,
      float transparentColor );

public:

   // ---------- Primitives ----------
   long addPrimitive( int type );
   void setPrimitive( 
      int   index, 
      float x, 
      float y, 
      float z, 
      float width, 
      float height, 
      int   martialId, 
      int   materialPadding );
   void rotatePrimitive( 
      int   index, 
      float x, 
      float y, 
      float z );
   void setPrimitiveMaterial( 
      int   index, 
      int   materialId ); 

public:

   // ---------- Complex objects ----------
   long addCube( 
      float x, 
      float y, 
      float z, 
      float radius, 
      int   martialId, 
      int   materialPadding );

public:

   // ---------- Lamps ----------
   long addLamp();
   void setLamp( 
      int index,
      float x, float y, float z, 
      float intensity, 
      float r, float g, float b );

public:

   // ---------- Materials ----------
   long addMaterial();
   void setMaterial( 
      int   index,
      float r, float g, float b, 
      float reflection, 
      float refraction,
      int   textured,
      float transparency,
      int   textureId,
      float specValue, float specPower, float specCoef,
      float innerIllumination );

public:

   // ---------- Camera ----------
   void setCamera( 
      cl_float4 eye, cl_float4 dir, cl_float4 angles );

public:

   // ---------- Textures ----------
   void setTexture(
      int   index,
      BYTE* texture );

   long addTexture( 
      const std::string& filename );

#ifdef USE_KINECT
public:

   // ---------- Kinect ----------

   long updateSkeletons( 
      double center_x, double  center_y, double  center_z, 
      double size,
      double radius,       int materialId,
      double head_radius,  int head_materialId,
      double hands_radius, int hands_materialId,
      double feet_radius,  int feet_materialId);
#endif // USE_KINECT

public:

   cl_int getNbActivePrimitives() { return m_nbActivePrimitives; };
   cl_int getNbActiveLamps()      { return m_nbActiveLamps; };
   cl_int getNbActiveMaterials()  { return m_nbActiveMaterials; };

public:

   int              getCLPlatformId() { return m_hPlatformId; };
   cl_context       getCLContext()    { return m_hContext; };
   cl_command_queue getCLQueue()      { return m_hQueue; };

private:

   char* loadFromFile( const std::string&, size_t&);

private:
   // OpenCL Objects
   cl_device_id     m_hDevices[100];
   int              m_hPlatformId;
   cl_context       m_hContext;
   cl_command_queue m_hQueue;
   cl_kernel        m_hKernel;
   cl_kernel        m_hKernelPostProcessing;
   cl_uint          m_computeUnits;
   cl_uint          m_preferredWorkGroupSize;

private:
   // Host
   cl_mem m_hBitmap;
   cl_mem m_hPrimitives;
   cl_mem m_hLamps;
   cl_mem m_hMaterials;
   cl_mem m_hVideo;
   cl_mem m_hDepth;
   cl_mem m_hTextures;
   cl_mem m_hRays;

   // Kinect declarations
#ifdef USE_KINECT
private:
   HANDLE             m_skeletons;
   HANDLE             m_hNextDepthFrameEvent; 
   HANDLE             m_hNextVideoFrameEvent;
   HANDLE             m_hNextSkeletonEvent;
   HANDLE             m_pVideoStreamHandle;
   HANDLE             m_pDepthStreamHandle;
   NUI_SKELETON_FRAME m_skeletonFrame;

   long               m_skeletonsBody;
   long               m_skeletonsLamp;

#endif // USE_KINECT

private:
   Primitive*  m_primitives;
   Lamp*       m_lamps;
   Material*   m_materials;
   cl_int      m_nbActivePrimitives;
   cl_int      m_nbActiveLamps;
   cl_int      m_nbActiveMaterials;
   cl_int      m_nbActiveTextures;
   cl_float4   m_viewPos;
   cl_float4   m_viewDir;
   cl_float4   m_angles;
   BYTE*       m_textures;
   bool        m_texturedTransfered;

private:
   cl_int      m_initialDraft;
   cl_int      m_draft;
};
