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

#include <math.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <sstream>

#define LOG_INFO( msg ) std::cout << msg << std::endl;
#define LOG_ERROR( msg ) std::cerr << msg << std::endl;

#include "OpenCLKernel.h"

const long MAX_SOURCE_SIZE = 65535;
const long MAX_DEVICES = 10;

/*
* getErrorDesc
*/
std::string getErrorDesc(int err)
{
   switch (err)
   {
   case CL_SUCCESS                        : return "CL_SUCCESS";
   case CL_DEVICE_NOT_FOUND               : return "CL_DEVICE_NOT_FOUND";
   case CL_COMPILER_NOT_AVAILABLE         : return "CL_COMPILER_NOT_AVAILABLE";
   case CL_MEM_OBJECT_ALLOCATION_FAILURE  : return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
   case CL_OUT_OF_RESOURCES               : return "CL_OUT_OF_RESOURCES";
   case CL_OUT_OF_HOST_MEMORY             : return "CL_OUT_OF_HOST_MEMORY";
   case CL_PROFILING_INFO_NOT_AVAILABLE   : return "CL_PROFILING_INFO_NOT_AVAILABLE";
   case CL_MEM_COPY_OVERLAP               : return "CL_MEM_COPY_OVERLAP";
   case CL_IMAGE_FORMAT_MISMATCH          : return "CL_IMAGE_FORMAT_MISMATCH";
   case CL_IMAGE_FORMAT_NOT_SUPPORTED     : return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
   case CL_BUILD_PROGRAM_FAILURE          : return "CL_BUILD_PROGRAM_FAILURE";
   case CL_MAP_FAILURE                    : return "CL_MAP_FAILURE";

   case CL_INVALID_VALUE                  : return "CL_INVALID_VALUE";
   case CL_INVALID_DEVICE_TYPE            : return "CL_INVALID_DEVICE_TYPE";
   case CL_INVALID_PLATFORM               : return "CL_INVALID_PLATFORM";
   case CL_INVALID_DEVICE                 : return "CL_INVALID_DEVICE";
   case CL_INVALID_CONTEXT                : return "CL_INVALID_CONTEXT";
   case CL_INVALID_QUEUE_PROPERTIES       : return "CL_INVALID_QUEUE_PROPERTIES";
   case CL_INVALID_COMMAND_QUEUE          : return "CL_INVALID_COMMAND_QUEUE";
   case CL_INVALID_HOST_PTR               : return "CL_INVALID_HOST_PTR";
   case CL_INVALID_MEM_OBJECT             : return "CL_INVALID_MEM_OBJECT";
   case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
   case CL_INVALID_IMAGE_SIZE             : return "CL_INVALID_IMAGE_SIZE";
   case CL_INVALID_SAMPLER                : return "CL_INVALID_SAMPLER";
   case CL_INVALID_BINARY                 : return "CL_INVALID_BINARY";
   case CL_INVALID_BUILD_OPTIONS          : return "CL_INVALID_BUILD_OPTIONS";
   case CL_INVALID_PROGRAM                : return "CL_INVALID_PROGRAM";
   case CL_INVALID_PROGRAM_EXECUTABLE     : return "CL_INVALID_PROGRAM_EXECUTABLE";
   case CL_INVALID_KERNEL_NAME            : return "CL_INVALID_KERNEL_NAME";
   case CL_INVALID_KERNEL_DEFINITION      : return "CL_INVALID_KERNEL_DEFINITION";
   case CL_INVALID_KERNEL                 : return "CL_INVALID_KERNEL";
   case CL_INVALID_ARG_INDEX              : return "CL_INVALID_ARG_INDEX";
   case CL_INVALID_ARG_VALUE              : return "CL_INVALID_ARG_VALUE";
   case CL_INVALID_ARG_SIZE               : return "CL_INVALID_ARG_SIZE";
   case CL_INVALID_KERNEL_ARGS            : return "CL_INVALID_KERNEL_ARGS";
   case CL_INVALID_WORK_DIMENSION         : return "CL_INVALID_WORK_DIMENSION";
   case CL_INVALID_WORK_GROUP_SIZE        : return "CL_INVALID_WORK_GROUP_SIZE";
   case CL_INVALID_WORK_ITEM_SIZE         : return "CL_INVALID_WORK_ITEM_SIZE";
   case CL_INVALID_GLOBAL_OFFSET          : return "CL_INVALID_GLOBAL_OFFSET";
   case CL_INVALID_EVENT_WAIT_LIST        : return "CL_INVALID_EVENT_WAIT_LIST";
   case CL_INVALID_OPERATION              : return "CL_INVALID_OPERATION";
   case CL_INVALID_GL_OBJECT              : return "CL_INVALID_GL_OBJECT";
   case CL_INVALID_BUFFER_SIZE            : return "CL_INVALID_BUFFER_SIZE";
   case CL_INVALID_MIP_LEVEL              : return "CL_INVALID_MIP_LEVEL";
   default: return "UNKNOWN";
   }
}

/*
* Callback function for clBuildProgram notifications
*/
void pfn_notify(cl_program, void *user_data)
{
   std::stringstream s;
   s << "OpenCL Error (via pfn_notify): " << user_data;
   std::cerr << s.str() << std::endl;
}

/*
* CHECKSTATUS
*/

#define CHECKSTATUS( stmt ) \
{ \
   int __status = stmt; \
   if( __status != CL_SUCCESS ) { \
   std::stringstream __s; \
   __s << "==> " #stmt "\n"; \
   __s << "ERROR : " << getErrorDesc(__status) << "\n" ; \
   __s << "<== " #stmt "\n"; \
   LOG_ERROR( __s.str() ); \
   } \
}

/*
* OpenCLKernel constructor
*/
OpenCLKernel::OpenCLKernel( int platformId, int deviceId, int nbWorkingItems, int draft )
 : m_hContext(0),m_hQueue(0),
   m_hBitmap(0), m_hVideo(0), m_hDepth(0), m_hTextures(0),
   m_hPrimitives(0), m_hLamps(0), m_primitives(0), m_lamps(0), m_materials(0),m_textures(0),
   m_nbActivePrimitives(0), m_nbActiveLamps(0),m_nbActiveMaterials(0),m_nbActiveTextures(0),
#if USE_KINECT
   m_skeletons(0), m_hNextDepthFrameEvent(0), m_hNextVideoFrameEvent(0), m_hNextSkeletonEvent(0),
   m_pVideoStreamHandle(0), m_pDepthStreamHandle(0),
   m_skeletonsBody(-1), m_skeletonsLamp(-1),
#endif // USE_KINECT
   m_computeUnits( nbWorkingItems ), m_preferredWorkGroupSize(0), m_initialDraft(draft), m_draft(1),
   m_texturedTransfered(false)
{
   int  status(0);
   cl_platform_id   platforms[MAX_DEVICES];
   cl_uint          ret_num_devices;
   cl_uint          ret_num_platforms;
   char buffer[MAX_SOURCE_SIZE];
   size_t len;

#if USE_KINECT
   // Initialize Kinect
   status = NuiInitialize( NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX | NUI_INITIALIZE_FLAG_USES_SKELETON | NUI_INITIALIZE_FLAG_USES_COLOR);

   m_hNextDepthFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
   m_hNextVideoFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
   m_hNextSkeletonEvent   = CreateEvent( NULL, TRUE, FALSE, NULL );

   m_skeletons = CreateEvent( NULL, TRUE, FALSE, NULL );			 
   status = NuiSkeletonTrackingEnable( m_skeletons, 0 );

   status = NuiImageStreamOpen( NUI_IMAGE_TYPE_COLOR,                  NUI_IMAGE_RESOLUTION_640x480, 0, 2, m_hNextVideoFrameEvent, &m_pVideoStreamHandle );
   status = NuiImageStreamOpen( NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX, NUI_IMAGE_RESOLUTION_320x240, 0, 2, m_hNextDepthFrameEvent, &m_pDepthStreamHandle );

   status = NuiCameraElevationSetAngle( 0 );
#endif // USE_KINECT


   LOG_INFO("clGetPlatformIDs\n");
   CHECKSTATUS(clGetPlatformIDs(MAX_DEVICES, platforms, &ret_num_platforms));
   //CHECKSTATUS(clGetPlatformIDs(0, NULL, &ret_num_platforms));

   std::stringstream s;
   //for( int p(0); p<ret_num_platforms; p++) 
   int p = platformId;
   {
      // Platform details
      s << "Platform " << p << ":\n";
      CHECKSTATUS(clGetPlatformInfo( platforms[p], CL_PLATFORM_PROFILE, MAX_SOURCE_SIZE, buffer, &len ));
      buffer[len] = 0; s << "  Profile    : " << buffer << "\n";
      CHECKSTATUS(clGetPlatformInfo( platforms[p], CL_PLATFORM_VERSION, MAX_SOURCE_SIZE, buffer, &len ));
      buffer[len] = 0; s << "  Version    : " << buffer << "\n";
      CHECKSTATUS(clGetPlatformInfo( platforms[p], CL_PLATFORM_NAME, MAX_SOURCE_SIZE, buffer, &len ));
      buffer[len] = 0; s << "  Name       : " << buffer << "\n";
      CHECKSTATUS(clGetPlatformInfo( platforms[p], CL_PLATFORM_VENDOR, MAX_SOURCE_SIZE, buffer, &len ));
      buffer[len] = 0; s << "  Vendor     : " << buffer << "\n";
      CHECKSTATUS(clGetPlatformInfo( platforms[p], CL_PLATFORM_VENDOR, MAX_SOURCE_SIZE, buffer, &len ));
      buffer[len] = 0; s << "  Extensions : " << buffer << "\n";

      CHECKSTATUS(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 1, m_hDevices, &ret_num_devices));

      // Devices
      int d = deviceId;
      //for( cl_uint d(0); d<ret_num_devices; d++ ) 
      {
         s << "  Device " << d << ":\n";

         CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
         s << "    DEVICE_NAME                        : " << buffer << "\n";
         CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
         s << "    DEVICE_VENDOR                      : " << buffer << "\n";
         CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));
         s << "    DEVICE_VERSION                     : " << buffer << "\n";
         CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL));
         s << "    DRIVER_VERSION                     : " << buffer << "\n";

         cl_uint value;
         cl_uint values[10];
         CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(value), &value, NULL));
         s << "    DEVICE_MAX_COMPUTE_UNITS           : " << value << "\n";
         CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(value), &value, NULL));
         s << "    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS : " << value << "\n";
         CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(value), &value, NULL));
         s << "    CL_DEVICE_MAX_WORK_GROUP_SIZE      : " << value << "\n";
         CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(values), &values, NULL));
         s << "    CL_DEVICE_MAX_WORK_ITEM_SIZES      : " << values[0] << ", " << values[1] << ", " << values[2] << "\n";
         CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(value), &value, NULL));
         s << "    CL_DEVICE_MAX_CLOCK_FREQUENCY      : " << value << "\n";


         cl_device_type infoType;
         CHECKSTATUS(clGetDeviceInfo(m_hDevices[d], CL_DEVICE_TYPE, sizeof(infoType), &infoType, NULL));
         s << "    DEVICE_TYPE                        : ";
         if (infoType & CL_DEVICE_TYPE_DEFAULT) {
            infoType &= ~CL_DEVICE_TYPE_DEFAULT;
            s << "Default";
         }
         if (infoType & CL_DEVICE_TYPE_CPU) {
            infoType &= ~CL_DEVICE_TYPE_CPU;
            s << "CPU";
         }
         if (infoType & CL_DEVICE_TYPE_GPU) {
            infoType &= ~CL_DEVICE_TYPE_GPU;
            s << "GPU";
         }
         if (infoType & CL_DEVICE_TYPE_ACCELERATOR) {
            infoType &= ~CL_DEVICE_TYPE_ACCELERATOR;
            s << "Accelerator";
         }
         if (infoType != 0) {
            s << "Unknown ";
            s << infoType;
         }
      }
      s << "\n";
   }
   std::cout << s.str() << std::endl;
   LOG_INFO( s.str() );

    m_hContext = clCreateContext(NULL, ret_num_devices, &m_hDevices[0], NULL, NULL, &status );

   m_hQueue = clCreateCommandQueue(m_hContext, m_hDevices[0], CL_QUEUE_PROFILING_ENABLE, &status);

   // Eye position
   m_viewPos.s[0] =   0.0f;
   m_viewPos.s[1] =   0.0f;
   m_viewPos.s[2] = -40.0f;

   // Rotation angles
   m_angles.s[0] = 0.0f;
   m_angles.s[1] = 0.0f;
   m_angles.s[2] = 0.0f;
}

/*
* compileKernels
*/
void OpenCLKernel::compileKernels( 
   const KernelSourceType sourceType,
   const std::string& source, 
   const std::string& ptxFileName,
   const std::string& options)
{
   try 
   {
      int status(0);
      cl_program hProgram(0);
      clUnloadCompiler();

      const char* source_str; 
      size_t len(0);
      switch( sourceType ) 
      {
      case kst_file:
         if( source.length() != 0 ) 
         {
            source_str = loadFromFile(source, len);
         }
         break;
      case kst_string:
         {
            source_str = source.c_str();
            len = source.length();
         }
         break;
      }


      LOG_INFO("clCreateProgramWithSource\n");
      hProgram = clCreateProgramWithSource( m_hContext, 1, (const char **)&source_str, (const size_t*)&len, &status );
      CHECKSTATUS(status);

      LOG_INFO("clCreateProgramWithSource\n");
      hProgram = clCreateProgramWithSource( m_hContext, 1, (const char **)&source_str, (const size_t*)&len, &status );
      CHECKSTATUS(status);

      LOG_INFO("clBuildProgram\n");
      CHECKSTATUS( clBuildProgram( hProgram, 0, NULL, options.c_str(), NULL, NULL) );
      
      if( sourceType == kst_file)
      {
         delete [] source_str;
         source_str = NULL;
      }

      clUnloadCompiler();

      LOG_INFO("clCreateKernel(render_kernel)\n");
      m_hKernel = clCreateKernel( hProgram, "render_kernel", &status );
      CHECKSTATUS(status);

      //if( m_computeUnits == 0 ) 
      {
         clGetKernelWorkGroupInfo( m_hKernel, m_hDevices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(m_computeUnits), &m_computeUnits , NULL);
         std::cout << "CL_KERNEL_WORK_GROUP_SIZE=" << m_computeUnits << std::endl;
      }

      clGetKernelWorkGroupInfo( m_hKernel, m_hDevices[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(m_preferredWorkGroupSize), &m_preferredWorkGroupSize , NULL);
      std::cout << "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE=" << m_preferredWorkGroupSize << std::endl;

      char buffer[MAX_SOURCE_SIZE];
      LOG_INFO("clGetProgramBuildInfo\n");
      CHECKSTATUS( clGetProgramBuildInfo( hProgram, m_hDevices[0], CL_PROGRAM_BUILD_LOG, MAX_SOURCE_SIZE*sizeof(char), &buffer, &len ) );

      if( buffer[0] != 0 ) 
      {
         buffer[len] = 0;
         std::stringstream s;
         s << buffer;
         LOG_INFO( s.str() );
         std::cout << s.str() << std::endl;
      }
#if 0
         // Generate Binaries!!!
         // Obtain the length of the binary data that will be queried, for each device
         size_t ret_num_devices = 1;
         size_t binaries_sizes[MAX_DEVICES];
         CHECKSTATUS( clGetProgramInfo(
            hProgram, 
            CL_PROGRAM_BINARY_SIZES, 
            ret_num_devices*sizeof(size_t), 
            binaries_sizes, 
            0 ));

         char **binaries = new char*[MAX_DEVICES];
         for (size_t i = 0; i < ret_num_devices; i++)
            binaries[i] = new char[binaries_sizes[i]+1];

         CHECKSTATUS( clGetProgramInfo(
            hProgram, 
            CL_PROGRAM_BINARIES, 
            MAX_DEVICES*sizeof(size_t), 
            binaries, 
            NULL));                        

         for (size_t i = 0; i < ret_num_devices; i++) {
            binaries[i][binaries_sizes[i]] = '\0';
            char name[255];
            sprintf_s(name, 255, "kernel%d.ptx", i );
            FILE* fp = NULL;
            fopen_s(&fp, name, "w");
            fwrite(binaries[i], 1, binaries_sizes[i], fp);
            fclose(fp);
         }

         for (size_t i = 0; i < ret_num_devices; i++)                                
            delete [] binaries[i];                        
         delete [] binaries;
#endif // 0

      if( ptxFileName.length() != 0 ) 
      {
         // Open the ptx file and load it   
         // into a char* buffer   
         std::ifstream myReadFile;
         std::string str;
         std::string line;
         std::ifstream myfile( ptxFileName.c_str() );
         if (myfile.is_open()) {
            while ( myfile.good() ) {
               std::getline(myfile,line);
               str += '\n' + line;
            }
            myfile.close();
         }

         size_t lSize = str.length();
         char* buffer = new char[lSize+1];
         strcpy_s( buffer, lSize, str.c_str() );

         // Build the rendering kernel
         int errcode(0);
         hProgram = clCreateProgramWithBinary(
            m_hContext,
            1, 
            &m_hDevices[0],
            &lSize, 
            (const unsigned char**)&buffer,                 
            &status, 
            &errcode);   
         CHECKSTATUS(errcode);

         CHECKSTATUS( clBuildProgram( hProgram, 0, NULL, "", NULL, NULL) );

         m_hKernel = clCreateKernel(
            hProgram, "render_kernel", &status );
         CHECKSTATUS(status);

         delete [] buffer;
      }

      LOG_INFO("clReleaseProgram\n");
      CHECKSTATUS(clReleaseProgram(hProgram));
      hProgram = 0;
   }
   catch( ... ) 
   {
      LOG_ERROR("Unexpected exception\n");
   }
}

void OpenCLKernel::initializeDevice(
   int        width, 
   int        height, 
   int        nbPrimitives,
   int        nbLamps,
   int        nbMaterials,
   int        nbTextures,
   BYTE*      bitmap)
{
   int status(0);
   // Setup device memory
   LOG_INFO("Setup device memory\n");
   m_hBitmap     = clCreateBuffer( m_hContext, CL_MEM_WRITE_ONLY, width*height*sizeof(BYTE)*gColorDepth,            0, NULL);

   m_hPrimitives = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , sizeof(Primitive)*nbPrimitives,                   0, NULL);
   m_hLamps      = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , sizeof(Lamp)*nbLamps,                             0, NULL);
   m_hMaterials  = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , sizeof(Material)*nbMaterials,                     0, NULL);

   m_hTextures   = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , gTextureWidth*gTextureHeight*gTextureDepth*sizeof(BYTE)*nbTextures, 0, NULL);

   m_hVideo      = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , gVideoWidth*gVideoHeight*gKinectColorVideo, 0, NULL);
   m_hDepth      = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY , gDepthWidth*gDepthHeight*gKinectColorDepth, 0, NULL);

   // Setup World
   m_primitives = new Primitive[nbPrimitives];
   memset( m_primitives, 0, nbPrimitives*sizeof(Primitive) ); 
   m_lamps      = new Lamp[nbLamps];
   memset( m_lamps, 0, nbLamps*sizeof(Lamp) ); 
   m_materials  = new Material[nbMaterials];
   memset( m_materials, 0, nbMaterials*sizeof(Material) ); 
   m_textures   = new BYTE[gTextureWidth*gTextureHeight*gColorDepth*nbTextures];

   // NVAPI
   /*
   StereoHandle * m_pStereoHandle = NULL;
   IUnknown *m_pDx = NULL;
   NvAPI_Stereo_CreateHandleFromIUnknown( m_pDx, m_pStereoHandle );
   NvAPI_Stereo_Activate( m_pStereoHandle );
   */
}

void OpenCLKernel::releaseDevice()
{
   LOG_INFO("Release device memory\n");
   if( m_hPrimitives ) CHECKSTATUS(clReleaseMemObject(m_hPrimitives));
   if( m_hLamps )      CHECKSTATUS(clReleaseMemObject(m_hLamps));
   if( m_hMaterials )  CHECKSTATUS(clReleaseMemObject(m_hMaterials));
   if( m_hTextures )   CHECKSTATUS(clReleaseMemObject(m_hTextures));

   if( m_hBitmap )     CHECKSTATUS(clReleaseMemObject(m_hBitmap));
   if( m_hVideo )      CHECKSTATUS(clReleaseMemObject(m_hVideo));
   if( m_hDepth )      CHECKSTATUS(clReleaseMemObject(m_hDepth));

   if( m_hKernel )     CHECKSTATUS(clReleaseKernel(m_hKernel));

   if( m_hQueue )      CHECKSTATUS(clReleaseCommandQueue(m_hQueue));
   if( m_hContext )    CHECKSTATUS(clReleaseContext(m_hContext));

   delete m_primitives;
   delete m_lamps;
   delete m_materials;
   delete m_textures;

   m_hContext=0;
   m_hQueue=0;
   m_hBitmap=0;
   m_hVideo=0;
   m_hDepth=0;
   m_hTextures=0;
   m_hPrimitives=0;
   m_hLamps=0;
   m_primitives=0;
   m_lamps=0;
   m_materials=0;
   m_textures=0;
   m_nbActivePrimitives=0;
   m_nbActiveLamps=0;
   m_nbActiveMaterials=0;
   m_nbActiveTextures=0;
#if USE_KINECT
   m_skeletons=0, 
   m_hNextDepthFrameEvent=0;
   m_hNextVideoFrameEvent=0;
   m_hNextSkeletonEvent=0;
   m_pVideoStreamHandle=0;
   m_pDepthStreamHandle=0;
   m_skeletonsBody = -1;
   m_skeletonsLamp = -1;
#endif // USE_KINECT
}

/*
* runKernel
*/
void OpenCLKernel::render( 
   int   width, 
   int   height, 
   BYTE* bitmap,
   float timer,
   float transparentColor)
{
   int status(0);

   BYTE* video(0);
   BYTE* depth(0);
#if USE_KINECT
   // Video
   const NUI_IMAGE_FRAME* pImageFrame = 0;
   WaitForSingleObject (m_hNextVideoFrameEvent,INFINITE); 
   status = NuiImageStreamGetNextFrame( m_pVideoStreamHandle, 0, &pImageFrame ); 
   if(( status == S_OK) && pImageFrame ) 
   {
      INuiFrameTexture* pTexture = pImageFrame->pFrameTexture;
      NUI_LOCKED_RECT LockedRect;
      pTexture->LockRect( 0, &LockedRect, NULL, 0 ) ; 
      if( LockedRect.Pitch != 0 ) 
      {
         video = (BYTE*) LockedRect.pBits;
      }
   }

   // Depth
   const NUI_IMAGE_FRAME* pDepthFrame = 0;
   WaitForSingleObject (m_hNextDepthFrameEvent,INFINITE); 
   status = NuiImageStreamGetNextFrame( m_pDepthStreamHandle, 0, &pDepthFrame ); 
   if(( status == S_OK) && pDepthFrame ) 
   {
      INuiFrameTexture* pTexture = pDepthFrame->pFrameTexture;
      if( pTexture ) 
      {
         NUI_LOCKED_RECT LockedRectDepth;
         pTexture->LockRect( 0, &LockedRectDepth, NULL, 0 ) ; 
         if( LockedRectDepth.Pitch != 0 ) 
         {
            depth = (BYTE*) LockedRectDepth.pBits;
         }
      }
   }
   NuiImageStreamReleaseFrame( m_pVideoStreamHandle, pImageFrame ); 
   NuiImageStreamReleaseFrame( m_pDepthStreamHandle, pDepthFrame ); 
#endif // USE_KINECT


   // Initialise Input arrays
   CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_hPrimitives, CL_FALSE, 0, m_nbActivePrimitives*sizeof(Primitive),                       m_primitives, 0, NULL, NULL));
   CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_hLamps,      CL_FALSE, 0, m_nbActiveLamps*sizeof(Lamp),                                 m_lamps,      0, NULL, NULL));
   CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_hMaterials,  CL_FALSE, 0, m_nbActiveMaterials*sizeof(Material),                         m_materials,  0, NULL, NULL));
   if( !m_texturedTransfered )
   {
      CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_hTextures,   CL_FALSE, 0, gTextureDepth*gTextureWidth*gTextureHeight*m_nbActiveTextures,m_textures,   0, NULL, NULL));
      m_texturedTransfered = true;
   }

   if( video ) CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_hVideo, CL_FALSE, 0, gKinectColorVideo*gVideoWidth*gVideoHeight, video, 0, NULL, NULL));
   if( depth ) CHECKSTATUS(clEnqueueWriteBuffer( m_hQueue, m_hDepth, CL_FALSE, 0, gKinectColorDepth*gDepthWidth*gDepthHeight, depth, 0, NULL, NULL));

   // Setting kernel arguments
   CHECKSTATUS(clSetKernelArg( m_hKernel, 0, sizeof(cl_float4),(void*)&m_viewPos ));
   CHECKSTATUS(clSetKernelArg( m_hKernel, 1, sizeof(cl_float4),(void*)&m_viewDir ));
   CHECKSTATUS(clSetKernelArg( m_hKernel, 2, sizeof(cl_float4),(void*)&m_angles ));
   CHECKSTATUS(clSetKernelArg( m_hKernel, 3, sizeof(cl_int),   (void*)&width ));
   CHECKSTATUS(clSetKernelArg( m_hKernel, 4, sizeof(cl_int),   (void*)&height ));
   CHECKSTATUS(clSetKernelArg( m_hKernel, 5, sizeof(cl_mem),   (void*)&m_hPrimitives ));
   CHECKSTATUS(clSetKernelArg( m_hKernel, 6, sizeof(cl_mem),   (void*)&m_hLamps ));
   CHECKSTATUS(clSetKernelArg( m_hKernel, 7, sizeof(cl_mem),   (void*)&m_hMaterials ));
   CHECKSTATUS(clSetKernelArg( m_hKernel, 8, sizeof(cl_int),   (void*)&m_nbActivePrimitives ));
   CHECKSTATUS(clSetKernelArg( m_hKernel, 9, sizeof(cl_int),   (void*)&m_nbActiveLamps ));
   CHECKSTATUS(clSetKernelArg( m_hKernel,10, sizeof(cl_int),   (void*)&m_nbActiveMaterials ));
   CHECKSTATUS(clSetKernelArg( m_hKernel,11, sizeof(cl_mem),   (void*)&m_hBitmap ));
   CHECKSTATUS(clSetKernelArg( m_hKernel,12, sizeof(cl_mem),   (void*)&m_hVideo ));
   CHECKSTATUS(clSetKernelArg( m_hKernel,13, sizeof(cl_mem),   (void*)&m_hDepth ));
   CHECKSTATUS(clSetKernelArg( m_hKernel,14, sizeof(cl_mem),   (void*)&m_hTextures ));
   CHECKSTATUS(clSetKernelArg( m_hKernel,15, sizeof(cl_float), (void*)&timer ));
   CHECKSTATUS(clSetKernelArg( m_hKernel,16, sizeof(cl_int),   (void*)&m_draft ));
   CHECKSTATUS(clSetKernelArg( m_hKernel,17, sizeof(cl_int),   (void*)&transparentColor ));

   // Run the kernel!!
   size_t szGlobalWorkSize[] = {width,height};
   size_t szLocalWorkSize  = 0;

   CHECKSTATUS(clEnqueueNDRangeKernel(
      m_hQueue, m_hKernel, 2, NULL, szGlobalWorkSize, 0, 0, 0, 0));

   // ------------------------------------------------------------
   // Read back the results
   // ------------------------------------------------------------

   // Bitmap
   if( bitmap != 0 ) {
      CHECKSTATUS( clEnqueueReadBuffer( m_hQueue, m_hBitmap, CL_FALSE, 0, width*height*sizeof(BYTE)*gColorDepth, bitmap, 0, NULL, NULL) );
   }

   CHECKSTATUS(clFlush(m_hQueue));
   CHECKSTATUS(clFinish(m_hQueue));

   m_draft--;
   m_draft = (m_draft < 1) ? 1 : m_draft;
}

void OpenCLKernel::setCamera( 
   cl_float4 eye, cl_float4 dir, cl_float4 angles )
{
   m_viewPos   = eye;
   m_viewDir   = dir;
   m_angles.s[0]  += angles.s[0];
   m_angles.s[1]  += angles.s[1];
   m_angles.s[2]  += angles.s[2];
   m_draft     = m_initialDraft;
}

/*
*
*/
OpenCLKernel::~OpenCLKernel()
{
   // Clean up
   releaseDevice();

#if USE_KINECT
   CloseHandle(m_skeletons);
   CloseHandle(m_hNextDepthFrameEvent); 
   CloseHandle(m_hNextVideoFrameEvent); 
   CloseHandle(m_hNextSkeletonEvent);
   NuiShutdown();
#endif // USE_KINECT
}

long OpenCLKernel::addPrimitive( int type )
{
   long result = m_nbActivePrimitives;
   m_primitives[m_nbActivePrimitives].type = type;
   m_primitives[m_nbActivePrimitives].materialId = NO_MATERIAL;
   m_nbActivePrimitives++;
   return result;
}

void OpenCLKernel::setPrimitive( 
   int index, 
   float x, 
   float y, 
   float z, 
   float width, 
   float height, 
   int   martialId, 
   int   materialPadding )
{
   if( index>= 0 && index < m_nbActivePrimitives) 
   {
      m_primitives[index].center.s[0]   = x;
      m_primitives[index].center.s[1]   = y;
      m_primitives[index].center.s[2]   = z;
      m_primitives[index].center.s[3]   = width; // Deprecated
      /*
      m_primitives[index].rotation.s[0] = 0.f;
      m_primitives[index].rotation.s[1] = 0.f;
      m_primitives[index].rotation.s[2] = 0.f;
      m_primitives[index].rotation.s[3] = 0.f; // Not used
      */
      m_primitives[index].size.s[0] = width;
      m_primitives[index].size.s[1] = height;
      m_primitives[index].size.s[2] = 0.f;
      m_primitives[index].size.s[3] = 0.f; // Not used
      m_primitives[index].materialId    = martialId;
      m_primitives[index].materialRatioX = (gTextureWidth/width/2)*materialPadding;
      m_primitives[index].materialRatioY = (gTextureHeight/height/2)*materialPadding;
   }
}

void OpenCLKernel::rotatePrimitive( 
   int   index, 
   float x, 
   float y, 
   float z )
{
   /*
   if( index>= 0 && index < m_nbActivePrimitives) 
   {
      m_primitives[index].rotation.s[0]   = x;
      m_primitives[index].rotation.s[1]   = y;
      m_primitives[index].rotation.s[2]   = z;
      m_primitives[index].rotation.s[3]   = 0.f; // Not used
   }
   */
}

long OpenCLKernel::addCube( 
   float x, 
   float y, 
   float z, 
   float radius, 
   int   martialId, 
   int   materialPadding )
{
   long returnValue;
   // Back
   returnValue = addPrimitive( ptXYPlane );
   setPrimitive( returnValue, x, y, z+radius, radius, radius, martialId, materialPadding ); 

   // Front
   returnValue = addPrimitive( ptXYPlane );
   setPrimitive( returnValue, x, y, z-radius, radius, radius, martialId, materialPadding ); 

   // Left
   returnValue = addPrimitive( ptYZPlane );
   setPrimitive( returnValue, x-radius, y, z, radius, radius, martialId, materialPadding ); 

   // Right
   returnValue = addPrimitive( ptYZPlane );
   setPrimitive( returnValue, x+radius, y, z, radius, radius, martialId, materialPadding ); 

   // Top
   returnValue = addPrimitive( ptXZPlane );
   setPrimitive( returnValue, x, y+radius, z, radius, radius, martialId, materialPadding ); 

   // Bottom
   returnValue = addPrimitive( ptXZPlane );
   setPrimitive( returnValue, x, y-radius, z, radius, radius, martialId, materialPadding ); 
   return returnValue;
}

void OpenCLKernel::setPrimitiveMaterial( 
   int   index, 
   int   materialId )
{
   if( index>= 0 && index < m_nbActivePrimitives) {
      m_primitives[index].materialId = materialId;
   }
}

long OpenCLKernel::addLamp()
{
   long result = m_nbActiveLamps;
   m_nbActiveLamps++;
   return result;
}

void OpenCLKernel::setLamp( 
   int index,
   float x, float y, float z, 
   float intensity, 
   float r, float g, float b )
{
   if( index>= 0 && index < m_nbActiveLamps ) {
      m_lamps[index].center.s[0]   = x;
      m_lamps[index].center.s[1]   = y;
      m_lamps[index].center.s[2]   = z;
      m_lamps[index].center.s[3]   = 0.5f; // radius
      m_lamps[index].color.s[0]    = r;
      m_lamps[index].color.s[1]    = g;
      m_lamps[index].color.s[2]    = b;
      m_lamps[index].color.s[3]    = intensity;
   }
}

// ---------- Materials ----------
long OpenCLKernel::addMaterial()
{
   long result = m_nbActiveMaterials;
   m_materials[m_nbActiveMaterials].textureId = NO_MATERIAL;
   m_nbActiveMaterials++;
   return result;
}

void OpenCLKernel::setMaterial( 
   int   index,
   float r, float g, float b, 
   float reflection, 
   float refraction, 
   int   textured,
   float transparency,
   int   textureId,
   float specValue, float specPower, float specCoef, float innerIllumination )
{
   if( index>= 0 && index < m_nbActiveMaterials ) {
      m_materials[index].color.s[0]  = r;
      m_materials[index].color.s[1]  = g;
      m_materials[index].color.s[2]  = b;
      m_materials[index].color.s[3]  = reflection;
      m_materials[index].refraction  = refraction;
      m_materials[index].textured    = textured;
      m_materials[index].textureId   = textureId;
      m_materials[index].transparency= transparency;
      m_materials[index].specular.s[0]  = specValue;
      m_materials[index].specular.s[1]  = specPower;
      m_materials[index].specular.s[2]  = innerIllumination;
      m_materials[index].specular.s[3]  = specCoef;
   }
}

// ---------- Textures ----------
void OpenCLKernel::setTexture(
   int   index,
   BYTE* texture )
{
   BYTE* idx = m_textures+index*gTextureWidth*gTextureHeight*gTextureDepth;
   int j(0);
   for( int i(0); i<gTextureWidth*gTextureHeight*gColorDepth; i += gColorDepth ) {
      idx[j]   = texture[i+2];
      idx[j+1] = texture[i+1];
      idx[j+2] = texture[i];
      j+=gTextureDepth;
   }
}

/*
*
*/
char* OpenCLKernel::loadFromFile( const std::string& filename, size_t& length )
{
   // Load the kernel source code into the array source_str
   FILE *fp = 0;
   char *source_str = 0;

   fopen_s( &fp, filename.c_str(), "r");
   if( fp == 0 ) 
   {
      std::cout << "Failed to load kernel " << filename.c_str() << std::endl;
   }
   else 
   {
      source_str = (char*)malloc(MAX_SOURCE_SIZE);
      length = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
      fclose( fp );
   }
   return source_str;
}

// ---------- Kinect ----------
long OpenCLKernel::addTexture( const std::string& filename )
{
   FILE *filePtr(0); //our file pointer
   BITMAPFILEHEADER bitmapFileHeader; //our bitmap file header
   unsigned char *bitmapImage;  //store image data
   BITMAPINFOHEADER bitmapInfoHeader;
   DWORD imageIdx=0;  //image index counter
   unsigned char tempRGB;  //our swap variable

   //open filename in read binary mode
   fopen_s(&filePtr, filename.c_str(), "rb");
   if (filePtr == NULL) {
      return 1;
   }

   //read the bitmap file header
   fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);

   //verify that this is a bmp file by check bitmap id
   if (bitmapFileHeader.bfType !=0x4D42) {
      fclose(filePtr);
      return 1;
   }

   //read the bitmap info header
   fread(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER),1,filePtr);

   //move file point to the begging of bitmap data
   fseek(filePtr, bitmapFileHeader.bfOffBits, SEEK_SET);

   //allocate enough memory for the bitmap image data
   bitmapImage = (unsigned char*)malloc(bitmapInfoHeader.biSizeImage);

   //verify memory allocation
   if (!bitmapImage)
   {
      free(bitmapImage);
      fclose(filePtr);
      return 1;
   }

   //read in the bitmap image data
   fread( bitmapImage, bitmapInfoHeader.biSizeImage, 1, filePtr);

   //make sure bitmap image data was read
   if (bitmapImage == NULL)
   {
      fclose(filePtr);
      return NULL;
   }

   //swap the r and b values to get RGB (bitmap is BGR)
   for (imageIdx = 0; imageIdx < bitmapInfoHeader.biSizeImage; imageIdx += 3)
   {
      tempRGB = bitmapImage[imageIdx];
      bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
      bitmapImage[imageIdx + 2] = tempRGB;
   }

   //close file and return bitmap iamge data
   fclose(filePtr);

   BYTE* index = m_textures + (m_nbActiveTextures*bitmapInfoHeader.biSizeImage*sizeof(BYTE));
   memcpy( index, bitmapImage, bitmapInfoHeader.biSizeImage );
   m_nbActiveTextures++;

   free( bitmapImage );
   return m_nbActiveTextures-1;
}

#ifdef USE_KINECT
long OpenCLKernel::updateSkeletons( 
   double center_x, double  center_y, double  center_z, 
   double size,
   double radius,       int materialId,
   double head_radius,  int head_materialId,
   double hands_radius, int hands_materialId,
   double feet_radius,  int feet_materialId)
{
   bool found = false;
   HRESULT hr = NuiSkeletonGetNextFrame( 0, &m_skeletonFrame );
   int i=0;
   while( i<NUI_SKELETON_COUNT && !found ) 
   {
      if( m_skeletonFrame.SkeletonData[i].eTrackingState == NUI_SKELETON_TRACKED ) 
      {
         found = true;
         if( m_skeletonsBody == -1 ) 
         {
            // Create Skeleton
            m_skeletonsBody = m_nbActivePrimitives;
            m_skeletonsLamp = m_nbActiveLamps;
            for( int j=0; j<20; j++ ) 
            {
               addPrimitive( ptSphere );
            }
         }
         else 
         {
            for( int j=0; j<20; j++ ) 
            {
               double r = radius;
               int   m = materialId;
               bool createSphere(true);
               switch (j) {
                  case NUI_SKELETON_POSITION_FOOT_LEFT:
                  case NUI_SKELETON_POSITION_FOOT_RIGHT:
                     r = feet_radius;
                     m = feet_materialId;
                     createSphere = true;
                     break;
                  case NUI_SKELETON_POSITION_HAND_LEFT:
                  case NUI_SKELETON_POSITION_HAND_RIGHT:
                     r = hands_radius;
                     m = hands_materialId;
                     createSphere = true;
                     /*
                     if( j == NUI_SKELETON_POSITION_HAND_RIGHT )
                     {
                        m_viewDir.s[0] = size*m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].x;
                        m_viewDir.s[1] = size*m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].y;
                        m_viewDir.s[2] = m_viewPos.s[2] + size*m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].z;
                     }
                     */
                     break;
                  case NUI_SKELETON_POSITION_HEAD:
                     r = head_radius;
                     m = head_materialId;
                     createSphere = true;
                     /*
                     m_viewPos.s[0] = size*m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].x;
                     m_viewPos.s[1] = size*m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].y;
                     m_viewPos.s[2] = size*m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].z - 400.f;
                     m_viewDir = m_viewPos;
                     */
                     break;
               }
               if( createSphere ) setPrimitive(
                  m_skeletonsBody+j,
                  static_cast<float>(m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].x * size + center_x),
                  static_cast<float>(m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].y * size + center_y),
                  static_cast<float>(m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].z * size + center_z),
                  static_cast<float>(r), 
                  static_cast<float>(r), 
                  m,
                  1 );
            }
#if 0
            m_angles.s[1] += 0.1f*asin( 
               m_skeletonFrame.SkeletonData[i].SkeletonPositions[NUI_SKELETON_POSITION_HAND_RIGHT].x - 
               m_skeletonFrame.SkeletonData[i].SkeletonPositions[NUI_SKELETON_POSITION_ELBOW_RIGHT].x );
            m_angles.s[0] += 0.1f*asin( 
               m_skeletonFrame.SkeletonData[i].SkeletonPositions[NUI_SKELETON_POSITION_HAND_LEFT].y - 
               m_skeletonFrame.SkeletonData[i].SkeletonPositions[NUI_SKELETON_POSITION_ELBOW_LEFT].y );
#endif // 0
            /*
            m_viewDir.s[2] += 200.f + 100.f*( 
               m_skeletonFrame.SkeletonData[i].SkeletonPositions[NUI_SKELETON_POSITION_HAND_RIGHT].z - 
               m_skeletonFrame.SkeletonData[i].SkeletonPositions[NUI_SKELETON_POSITION_HAND_LEFT].z );
               */
         }
      }
      i++;
   }
   return 0;
}
#endif // USE_KINECT
