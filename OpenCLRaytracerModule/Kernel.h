#ifndef _KERNEL_H_
#define _KERNEL_H_

extern "C" void kernel_setTextureFilterMode(bool bLinearFilter);

#if USE_CUDA
extern "C" void kernel_initCuda( 
   BYTE*      bitmap, 
   cudaExtent bitmapSize, 
   int        width, 
   int        height, 
   int        nbPrimitives,
   int        nbLamps,
   BYTE*      textures,
   int        nbTextures );
#endif // USE_CUDA

extern "C" void kernel_finalizeCuda();

extern "C" void kernel_updateVideo( BYTE* video );

extern "C" void kernel_setCamera( float3 eye, int width, int height );

// Primitives
extern "C" void kernel_addPrimitive( 
   int type, 
   float x, float y, float z, float radius, 
   float reflection, float r, float g, float b, 
   bool textured, bool transparent, int materialId, int& index );

extern "C" void kernel_setPrimitive( 
   int   index, 
   float x, float y, float z, float radius, 
   float reflection, float r, float g, float b, 
   bool  textured, bool transparent, int materialId );

// Lamps
extern "C" void kernel_addLamp( 
   float x, float y, float z, 
   float intensity, 
   float r, float g, float b );

extern "C" void kernel_render(
   int   imageW, 
   int   imageH, 
   BYTE* bitmap,
   float time,
   bool  viewHasChanged );

#endif // _KERNEL_H_
