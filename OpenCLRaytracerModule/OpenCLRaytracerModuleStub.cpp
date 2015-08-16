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

#include <windows.h>
#include "OpenCLRaytracerModuleStub.h"

#include <fstream>

#include "OpenCLKernel.h"
OpenCLKernel* oclKernel = 0;

// Global variables
int    gImageWidth     =  0;
int    gImageHeight    =  0;
bool   gViewHasChanged =  true;
BYTE*  gRenderBitmap   =  0;
double gTime           =  0.0f;

// Gesture
cl_float gAngleX = 0.f;
cl_float gAngleY = 0.f;
cl_float gAngleZ = 0.f;
cl_float gDistance = 0.f;

// Textures
cl_float4  gEye;
cl_float4  gDir;

// --------------------------------------------------------------------------------
// Forward declarations
// --------------------------------------------------------------------------------
extern "C" void setTextureFilterMode(
   bool bLinearFilter);

// --------------------------------------------------------------------------------
// Implementation
// --------------------------------------------------------------------------------
extern "C" OPENCLRAYTRACERMODULE_API 
   long RayTracer_CreateScene(
   int     platformId,
   int     deviceId,
   char*   kernelCode,
   int     width,
   int     height,
   int     nbPrimitives,
   int     nbLamps,
   int     nbMaterials,
   int     nbTextures,
   int     nbWorkingItems,
   HANDLE& display,
   HANDLE& kinect)
{
   gImageWidth   = width;
   gImageHeight  = height;
   gRenderBitmap = new BYTE[width*height*gColorDepth];
   gTime = 0.f;

   oclKernel = new OpenCLKernel( platformId, deviceId, nbWorkingItems, 1 );
   oclKernel->compileKernels( kst_string, kernelCode, "", "" );
   oclKernel->initializeDevice( width, height, nbPrimitives, nbLamps, nbMaterials, nbTextures, gRenderBitmap );

   gViewHasChanged = true;

   display  = gRenderBitmap; // Returns the rended bitmap to the caller
   return 0;
}

// --------------------------------------------------------------------------------
extern "C" OPENCLRAYTRACERMODULE_API 
   long RayTracer_DeleteScene()
{
   // kernel_finalizeOPENCL();
   if( gRenderBitmap ) delete [] gRenderBitmap;
   if( oclKernel ) delete oclKernel;
   return 0;   
}

// --------------------------------------------------------------------------------
extern "C" OPENCLRAYTRACERMODULE_API 
   void RayTracer_SetCamera( 
   double eye_x,   double eye_y,   double eye_z,
   double dir_x,   double dir_y,   double dir_z,
   double angle_x, double angle_y, double angle_z )
{
   cl_float4 eye;
   eye.s[0] = static_cast<cl_float>(eye_x);
   eye.s[1] = static_cast<cl_float>(eye_y);
   eye.s[2] = static_cast<cl_float>(eye_z + gDistance);

   gDir.s[0] = static_cast<cl_float>(dir_x);
   gDir.s[1] = static_cast<cl_float>(dir_y);
   gDir.s[2] = static_cast<cl_float>(dir_z);

   cl_float4 angles;
   angles.s[0] = static_cast<cl_float>(angle_x + gAngleX);
   angles.s[1] = static_cast<cl_float>(angle_y + gAngleY);
   angles.s[2] = static_cast<cl_float>(angle_z + gAngleZ);

   oclKernel->setCamera( eye, gDir, angles );
}

// --------------------------------------------------------------------------------
extern "C" OPENCLRAYTRACERMODULE_API 
   long RayTracer_RunKernel( double timer, double transparentColor )
{
   oclKernel->render( 
      gImageWidth, gImageHeight, 
      gRenderBitmap,
      static_cast<cl_float>(gTime),
      static_cast<cl_float>(transparentColor) );
   gTime += 0.1f;
   return 0;
}

// --------------------------------------------------------------------------------
extern "C" OPENCLRAYTRACERMODULE_API 
   long RayTracer_AddPrimitive( int type )
{
   return oclKernel->addPrimitive( type );
}

// --------------------------------------------------------------------------------
extern "C" OPENCLRAYTRACERMODULE_API 
   long RayTracer_SetPrimitive( 
   int    index,
   double center_x, 
   double center_y, 
   double center_z, 
   double width,
   double height,
   int    materialId, 
   int    materialPadding )
{
   oclKernel->setPrimitive( 
      index, 
      static_cast<cl_float>(center_x), 
      static_cast<cl_float>(center_y), 
      static_cast<cl_float>(center_z), 
      static_cast<cl_float>(width), 
      static_cast<cl_float>(height), 
      materialId, materialPadding );
   return 0;
}

extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_RotatePrimitive( 
   int    index,
   double center_x, 
   double center_y, 
   double center_z)
{
   oclKernel->rotatePrimitive( 
      index, 
      static_cast<cl_float>(center_x), 
      static_cast<cl_float>(center_y), 
      static_cast<cl_float>(center_z));
   return 0;
}

extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_SetPrimitiveMaterial( 
   int    index,
   int    materialId)
{
   oclKernel->setPrimitiveMaterial( 
      index, 
      materialId);
   return 0;
}


// --------------------------------------------------------------------------------
extern "C" OPENCLRAYTRACERMODULE_API 
   long RayTracer_AddLamp()
{
   return oclKernel->addLamp();
}

// --------------------------------------------------------------------------------
extern "C" OPENCLRAYTRACERMODULE_API 
   long RayTracer_SetLamp( 
   int     index,
   double  center_x,
   double  center_y,
   double  center_z,
   double  intensity,
   double  color_r,
   double  color_g,
   double  color_b )
{
   oclKernel->setLamp(
      index,
      static_cast<cl_float>(center_x), static_cast<cl_float>(center_y), static_cast<cl_float>(center_z), 
      static_cast<cl_float>(intensity), 
      static_cast<cl_float>(color_r), static_cast<cl_float>(color_g), static_cast<cl_float>(color_b) );
   return 0;
}

// --------------------------------------------------------------------------------
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_UpdateSkeletons( 
   double center_x, double  center_y, double center_z, 
   double size,
   double radius,       int materialId,
   double head_radius,  int head_materialId,
   double hands_radius, int hands_materialId,
   double feet_radius,  int feet_materialId)
{
#if USE_KINECT
   return oclKernel->updateSkeletons(
      center_x, center_y, center_z, 
      size,
      radius,       materialId,
      head_radius,  head_materialId,
      hands_radius, hands_materialId,
      feet_radius,  feet_materialId);
#else
   return 0;
#endif // USE_KINECT
}

// --------------------------------------------------------------------------------
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_AddTexture( char* filename )
{
   return oclKernel->addTexture( filename );
}

// --------------------------------------------------------------------------------
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_SetTexture( int index, HANDLE texture )
{
   oclKernel->setTexture( 
      index, 
      static_cast<BYTE*>(texture) );
   return 0;
}

// ---------- Materials ----------
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_AddMaterial()
{
   return oclKernel->addMaterial();
}

extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_SetMaterial(
   int    index,
   double color_r, 
   double color_g, 
   double color_b,
   double reflection,
   double refraction,
   int    textured,
   float  transparency,
   int    textureId,
   double specValue, 
   double specPower, 
   double specCoef,
   double innerIllumination)
{
   oclKernel->setMaterial( 
      index, 
      static_cast<cl_float>(color_r), 
      static_cast<cl_float>(color_g), 
      static_cast<cl_float>(color_b), 
      static_cast<cl_float>(reflection), 
      static_cast<cl_float>(refraction),
      textured, 
      static_cast<cl_float>(transparency), 
      textureId,
      static_cast<cl_float>(specValue),
      static_cast<cl_float>(specPower), 
      static_cast<cl_float>(specCoef ),
      static_cast<cl_float>(innerIllumination));
   return 0;
}
