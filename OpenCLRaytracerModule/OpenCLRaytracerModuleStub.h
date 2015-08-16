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

#include <windows.h>
#include "OpenCLKernel.h"

// ---------- Scene ----------
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_CreateScene(
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
   HANDLE& bitmap,
   HANDLE& kinect);
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_DeleteScene();

// ---------- Camera ----------
extern "C" OPENCLRAYTRACERMODULE_API void RayTracer_SetCamera( 
   double eye_x,   double eye_y,   double eye_z,
   double dir_x,   double dir_y,   double dir_z,
   double angle_x, double angle_y, double angle_z);

// ---------- Rendering ----------
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_RunKernel( double timer, double transparentColor );

// ---------- Primitives ----------
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_AddPrimitive( int type );
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_SetPrimitive( 
   int    index,
   double center_x, 
   double center_y, 
   double center_z, 
   double width,
   double height,
   int    materialId, 
   int    materialPadding);
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_RotatePrimitive( 
   int    index,
   double center_x, 
   double center_y, 
   double center_z);
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_SetPrimitiveMaterial( 
   int    index,
   int    materialId);

// ---------- Lamps ----------
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_AddLamp();
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_SetLamp( 
   int     index,
   double  center_x, 
   double  center_y, 
   double  center_z,
   double  intensity,
   double  color_r, 
   double  color_g, 
   double  color_b );

// ---------- Materials ----------
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_AddMaterial();
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
   double innerIllumination);

// ---------- Textures ----------
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_AddTexture( char* filename );
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_SetTexture( int index, HANDLE texture );

// ---------- Kinect ----------
extern "C" OPENCLRAYTRACERMODULE_API long RayTracer_UpdateSkeletons(
   double center_x, double  center_y, double  center_z, 
   double size,
   double radius,       int materialId,
   double head_radius,  int head_materialId,
   double hands_radius, int hands_materialId,
   double feet_radius,  int feet_materialId);
