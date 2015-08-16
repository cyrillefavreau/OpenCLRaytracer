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

// OpenGL Graphics Includes
#include <GL/glew.h>
#include <GL/freeglut.h>

// Includes
#include <memory>
#include <time.h>
#include <iostream>
#include <cassert>

#include "../OpenCLRaytracerModule/OpenCLKernel.h"

#include "kernel.h"

// General Settings
const long REFRESH_DELAY = 10; //ms
const bool gUseKinect    = false;

// Rendering window vars
const unsigned int draft        = 1;
unsigned int window_width       = 512;
unsigned int window_height      = window_width*9/16;
const unsigned int window_depth = 4;

// Scene
const float gRoomSize = 500.f;
const int nbSlices = 16;
int currentMaterial = 0;


// Scene
int platform     = 2;
int device       = 0;
int nbPrimitives = 0;
int nbLamps      = 0;
int nbMaterials  = 0;
int nbTextures   = 0;
float transparentColor = 0.5f;

// Camera
cl_float4 eye;
cl_float4 direction;
cl_float4 angles;

// sphere
cl_float4 activeSphereCenter[3]    = {{0.f, 100.f, 0.f, 50.f},{0.f, 50.f, 0.f, 90.f},{0.f, 10.f, 0.f, 60.f}};
cl_float4 activeSphereDirection[3] = {{1.4f, 0.f, 1.3f, 0.f},{-0.2f, 0.f, 0.6f, 0.f},{ 0.2f, 0.f, -0.18f, 0.f}};
int       activeSphereId           = -1;
int       activeSphereMaterial[3]  = {rand()%20,rand()%20,rand()%20};

// materials
float reflection = 0.f;
float refraction = 1.f;

#ifdef USE_KINECT
const float gSkeletonSize      = 200.0;
const float gSkeletonThickness = 20.0;
#endif // USE_KINECT

// OpenGL
GLubyte* ubImage;
int previousFps=0;

/**
--------------------------------------------------------------------------------
OpenGL
--------------------------------------------------------------------------------
*/

// GL functionality
void initgl(int argc, char** argv);
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion( int x, int y );
void timerEvent( int value );
void createScene( int platform, int device );

// Helpers
void TestNoGL();
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

// Sim and Auto-Verification parameters 
float anim = 0.0;
bool bNoPrompt = false;  

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

// Raytracer Module
OpenCLKernel* oclKernel = 0;
unsigned int* uiOutput = NULL;

float getRandomValue( int range, int safeZone, bool allowNegativeValues = true )
{
   float value( static_cast<float>(rand()%range) + safeZone);
   if( allowNegativeValues ) 
   {
      value *= (rand()%2==0)? -1 : 1;
   }
   return value;
}

void idle()
{
}

void cleanup()
{
}

/*
--------------------------------------------------------------------------------
setup the window and assign callbacks
--------------------------------------------------------------------------------
*/
void initgl( int argc, char **argv )
{
   size_t len(window_width*window_height*window_depth);
   ubImage = new GLubyte[len];
   memset( ubImage, 0, len ); 

   glutInit(&argc, (char**)argv);
   glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );

   glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2, glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);
   
   glutInitWindowSize(window_width, window_height);
   glutCreateWindow("OpenCL Raytracer");
   
   glutDisplayFunc(display);       // register GLUT callback functions
   glutKeyboardFunc(keyboard);
   glutMouseFunc(mouse);
   glutMotionFunc(motion);
   glutTimerFunc(REFRESH_DELAY,timerEvent,1);
   return;
}

void TexFunc(void)
{
  glEnable(GL_TEXTURE_2D);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

  glTexImage2D(GL_TEXTURE_2D, 0, 3, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, ubImage);

  glBegin(GL_QUADS);
  glTexCoord2f(1.0, 1.0);
  glVertex3f(-1.0, 1.0, 0.0);
  glTexCoord2f(0.0, 1.0);
  glVertex3f( 1.0, 1.0, 0.0);
  glTexCoord2f(0.0, 0.0);
  glVertex3f( 1.0,-1.0, 0.0);
  glTexCoord2f(1.0, 0.0);
  glVertex3f(-1.0,-1.0, 0.0);
  glEnd();

  glDisable(GL_TEXTURE_2D);
}

// Display callback
//*****************************************************************************
void display()
{
   // clear graphics
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   char text[255];
   long t = GetTickCount();
   oclKernel->render( window_width, window_height, (BYTE*)ubImage, anim, transparentColor );
   t = GetTickCount()-t;
   sprintf_s(text, "OpenCL Raytracer (%d Fps)", 1000/((t+previousFps)/2));
   previousFps = t;
   
   TexFunc();
   glutSetWindowTitle(text);
   glFlush();

   glutSwapBuffers();
}

void timerEvent(int value)
{
#if USE_KINECT
    oclKernel->updateSkeletons(
       0.0, gSkeletonSize-200, -150.0,          // Position
       gSkeletonSize,                // Skeleton size
       gSkeletonThickness,        0, // Default size and material
       gSkeletonThickness*2.0f,  10, // Head size and material
       gSkeletonThickness*1.5f,   1, // Hands size and material
       gSkeletonThickness*1.8f,  10  // Feet size and material
    );
#endif // USE_KINEXT

#if 0
   //oclKernel->rotatePrimitive( 2, 10.f*cos(anim), 10.f*sin(anim), 0.f );
   for( int i(0); i<3; ++i )
   {
      activeSphereCenter[i].s[0] += activeSphereDirection[i].s[0];
      activeSphereCenter[i].s[1]  = -200 + activeSphereCenter[i].s[3] + ((i==0) ? 0 : activeSphereCenter[i].s[3]*fabs(cos(anim/2.f+i/2.f)));
      activeSphereCenter[i].s[2] += activeSphereDirection[i].s[2];
      oclKernel->setPrimitive( 
         activeSphereId+i, 
         activeSphereCenter[i].s[0], activeSphereCenter[i].s[1], activeSphereCenter[i].s[2], 
         activeSphereCenter[i].s[3], activeSphereCenter[i].s[3], 
         activeSphereMaterial[i], 1 ); 

      if( fabs(activeSphereCenter[i].s[0]) > (gRoomSize-activeSphereCenter[i].s[3])) activeSphereDirection[i].s[0] = -activeSphereDirection[i].s[0];
      if( fabs(activeSphereCenter[i].s[2]) > (gRoomSize-activeSphereCenter[i].s[3])) activeSphereDirection[i].s[2] = -activeSphereDirection[i].s[2];
   }
#endif // 0

   //oclKernel->setLamp( 0, 800*cos(anim/50.f), 800, 800*sin(anim/50.f), 1.f, 1.f, 1.f, 1.0f );

   //oclKernel->setMaterial(0, 1.f, 1.f, 1.f, 0.9f, 1.f+0.001f*cos(anim*0.1f), 0, 0, NO_MATERIAL, 10.f, 100.f, 10.f, 0.f );

   glutPostRedisplay();
   glutTimerFunc(REFRESH_DELAY, timerEvent,0);

   //angles.s[1] = 0.01f;
   //angles.s[0] = sin(anim/100.f)/100.f;
   //oclKernel->setCamera( eye, direction, angles );
#if 0
   if( int(anim*10)%100 == 0 ) 
   {
      oclKernel->setPrimitiveMaterial( rand()%nbPrimitives, rand()%nbMaterials );
   }
#endif // 0

   anim += 0.5f;
}

// Keyboard events handler
//*****************************************************************************
void keyboard(unsigned char key, int x, int y)
{
   srand(static_cast< unsigned int>(time(NULL))); 

   switch(key) 
   {
   case 'R':    
   case 'r':
      {
         // Reset scene
         delete oclKernel;
         oclKernel = 0;
         createScene( platform, device );
         break;
      }
   case 'F':
   case 'f':
      {
         // Toggle to full screen mode
         glutFullScreen();
         break;
      }

   case 'C':
   case 'c':
      {
         cl_float4 pos;
         pos.s[0] = getRandomValue( static_cast<int>(gRoomSize/2.f), 0 );
         pos.s[1] = getRandomValue( static_cast<int>(gRoomSize/2.f), 0, false ) - 100.f; 
         pos.s[2] = getRandomValue( static_cast<int>(gRoomSize/2.f), 0 );
         pos.s[3] = getRandomValue( 100, 10, false );
         int m  = rand()%nbMaterials;
         nbPrimitives = oclKernel->addCube( pos.s[0], pos.s[1], pos.s[2], pos.s[3], m, 1 );
         std::cout << "Cube added: " << nbPrimitives+1 << " primitives" << std::endl;
         break;
      }

   case 'S':
   case 's':
      {
         // Add sphere
         float r = getRandomValue( 100,  50, false );
         nbPrimitives = oclKernel->addPrimitive( ptSphere );
         oclKernel->setPrimitive(
            nbPrimitives,
            getRandomValue( static_cast<int>(gRoomSize/2.f),   static_cast<int>(gRoomSize/4.f) ), 
            getRandomValue( static_cast<int>(gRoomSize/2.f),   static_cast<int>(gRoomSize/4.f), false ) - 200,
            getRandomValue( static_cast<int>(gRoomSize/2.f),   static_cast<int>(gRoomSize/4.f) ),
            r, r,
            rand()%20, 1 );
         std::cout << "Sphere added: " << nbPrimitives+1 << " primitives" << std::endl;
         break;
      }

   case 'Y':
   case 'y':
      {
         // Add Cylinder
         nbPrimitives = oclKernel->addPrimitive( ptCylinder );
         float r = getRandomValue( 100,  50, false );
         oclKernel->setPrimitive(
            nbPrimitives,
            getRandomValue( static_cast<int>(gRoomSize/2.f),   static_cast<int>(gRoomSize/4.f) ), 
            getRandomValue( static_cast<int>(gRoomSize/2.f),   static_cast<int>(gRoomSize/4.f), false ) -200,
            getRandomValue( static_cast<int>(gRoomSize/2.f),   static_cast<int>(gRoomSize/4.f) ),
            r, r, 
            rand()%20, 1 );
         std::cout << "Cylinder added: " << nbPrimitives+1 << " primitives" << std::endl;
         break;
      }

   case 'P':
   case 'p':
      {
         // Add plan
         float r = getRandomValue( 200, 50, false );
         float x = getRandomValue( static_cast<int>(gRoomSize), 0 );
         float y = getRandomValue( 200, 0 );
         float z = getRandomValue( static_cast<int>(gRoomSize), 0 );
         int   m = 10+rand()%10;
         nbPrimitives = oclKernel->addPrimitive( ptXYPlane );
         oclKernel->setPrimitive( nbPrimitives, x, y, z, r, r, m, 1 );
         nbPrimitives = oclKernel->addPrimitive( ptXYPlane );
         oclKernel->setPrimitive( nbPrimitives, x, y, z, r, r, m, 1 );
         std::cout << "Plan added: " << nbPrimitives+1 << " primitives" << std::endl;
         break;
      }

   case 'e':
      {
         transparentColor += 0.01f;
         if( transparentColor>0.99f) transparentColor=0.99f;
         break;
      }
   case 'd':
      {
         transparentColor -= 0.01f;
         if( transparentColor<0.f) transparentColor=0.f;
         break;
      }
   case 'L':
   case 'l':
      {
         // Add lamp
         nbLamps = oclKernel->addLamp();
         oclKernel->setLamp(
            nbLamps,
            getRandomValue( 1000, 0 ), 
            200+getRandomValue( 100, 0 ), 
            getRandomValue( 1000, 0 ),
            0.1f, 
            rand()%100/100.f ,
            rand()%100/100.f , 
            rand()%100/100.f );
         std::cout << nbLamps+1 << " lamps" << std::endl;
         break;
      }

   case 'M':
   case 'm':
      {
         for( int i(0); i<nbSlices; ++i)
         {
            oclKernel->setPrimitiveMaterial( (i*3)+5, i+20 );
            oclKernel->setPrimitiveMaterial( (i*3)+6, i+20+nbSlices );
            oclKernel->setPrimitiveMaterial( (i*3)+7, i+20+nbSlices*2 );
         }
         break;
      }

   case 'N':
   case 'n':
      {
         for( int i(0); i<nbPrimitives-5; ++i)
         {
            oclKernel->setPrimitiveMaterial( i+5, currentMaterial%nbMaterials );
         }
         currentMaterial++;
         break;
      }

   case '\033': 
   case '\015': 
   case 'X':    
   case 'x':    
      {
         // Cleanup up and quit
         bNoPrompt = true;
         Cleanup(EXIT_SUCCESS);
         break;
      }
   }
}

// Mouse event handlers
//*****************************************************************************
void mouse(int button, int state, int x, int y)
{
   if (state == GLUT_DOWN) 
   {
      mouse_buttons |= 1<<button;
   } 
   else 
   {
      if (state == GLUT_UP) 
      {
         mouse_buttons = 0;
      }
   }
   mouse_old_x = x;
   mouse_old_y = y;
   angles.s[0] = 0.f;
   angles.s[1] = 0.f;
}

void motion(int x, int y)
{
   switch( mouse_buttons ) 
   {
   case 1:
      // Move eye position along the Z axis
      eye.s[2] += 2*(mouse_old_y-y);
      if( glutGetModifiers() != GLUT_ACTIVE_SHIFT ) 
      {
         direction.s[2] += 2*(mouse_old_y-y);
      }
      break;
   case 2:
      // Rotates the scene around X and Y axis
      angles.s[1] = -asin( (mouse_old_x-x) / 100.f );
      angles.s[0] = asin( (mouse_old_y-y) / 100.f );
      break;
   case 4:
      // Move eye postion along X and Y axis
      eye.s[0]       += (mouse_old_x-x);
      eye.s[1]       += (mouse_old_y-y);
      direction.s[0] += (mouse_old_x-x);
      direction.s[1] += (mouse_old_y-y);
      break;
   }
   mouse_old_x = x;
   mouse_old_y = y;
   oclKernel->setCamera( eye, direction, angles );
}

// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
   // Cleanup allocated objects
   std::cout << "\nStarting Cleanup...\n\n" << std::endl;
   if( ubImage ) delete [] ubImage;
   delete oclKernel;

   exit (iExitCode);
}

void createMaterials()
{
   // Materials
   for( int i(0); i<20+(nbSlices*3); ++i ) 
   {
      float reflection = 0.f;
      float refraction = 0.f;
      int   texture = NO_MATERIAL;
      float transparency = 0.f;
      int   procedural = 0;

      switch( i%10 ) 
      {
      case 0:
         {
            reflection = (i==0) ? 0.f : rand()%100/100.f;
            break;
         }
      case 1:
         {
            // 10 to 19 - Refraction without transparency
            transparency = 0.95f;
            //reflection   = 0.9f;
            refraction   = 1.33f;
            break;
         }

      default:
         {
            // 20 to 29 - Refraction with transparency
            //if( (i-20)>=nbSlices ) 
            transparency = 0.1f;
            //reflection   = 0.1f;
            //refraction   = 1.0f+rand()%100/1000.f;
            texture      = i-20;
            break;
         }
      }

      nbMaterials = oclKernel->addMaterial();
      oclKernel->setMaterial(
         nbMaterials,
         rand()%100/100.f, 
         rand()%100/100.f, 
         rand()%100/100.f,
         reflection, refraction,
         procedural,   
         transparency,
         texture,
         0.5, 200.0, 1.0,
         0.f);
   }
   std::cout << nbMaterials+1 << " materials" << std::endl;
}

void createTextures()
{
   // Textures
   // XZ
   for( int i(0); i<nbSlices; i++)
   {
      int index = i*(166/nbSlices);
      char tmp[5];
      sprintf_s(tmp, "%3d", index);
      std::string filename("../Textures/Desktops/");
      filename += tmp;
      filename += ".bmp";
      nbTextures = oclKernel->addTexture(filename.c_str());
   }
   std::cout << nbTextures+1 << " textures" << std::endl;
}

void createScene( int platform, int device )
{
   nbPrimitives = 0;
   nbLamps      = 0;
   nbMaterials  = 0;
   nbTextures   = 0;
   srand(static_cast< unsigned int>(time(NULL))); 

   oclKernel = new OpenCLKernel( platform, device, 128, draft );
   oclKernel->initializeDevice( window_width, window_height, 512, 32, 20+(nbSlices+1)*3, (nbSlices+1)*3, NULL );
   oclKernel->compileKernels( kst_file, "../OpenCLRaytracerModule/Kernel.cl", "", "-cl-fast-relaxed-math" );

   eye.s[0] =    0.f;
   eye.s[1] =    0.f;
   eye.s[2] = -400.f;

   direction.s[0] = 0.f;
   direction.s[1] = 0.f;
   direction.s[2] = 0.f;

   angles.s[0] = 0.f;
   angles.s[1] = 0.f;
   angles.s[2] = 0.f;

   oclKernel->setCamera( eye, direction, angles );

   createMaterials();

   // Checkboard
   nbPrimitives = oclKernel->addPrimitive( ptCheckboard );
   oclKernel->setPrimitive( nbPrimitives, 0.0, -200.0, 5.f, gRoomSize, gRoomSize, 0, 1 ); 

   // Sphere
   nbPrimitives = oclKernel->addPrimitive( ptSphere );
   oclKernel->setPrimitive( nbPrimitives, -100.f, 0.f, 0.f, 200.f, 0.f, 1, 1 ); 
   nbPrimitives = oclKernel->addPrimitive( ptSphere );
   oclKernel->setPrimitive( nbPrimitives,  100.f, 0.f, 0.f, 200.f, 0.f, 1, 1 ); 
   nbPrimitives = oclKernel->addPrimitive( ptSphere );
   oclKernel->setPrimitive( nbPrimitives, 0.f, 100.f,  0.f, 200.f, 0.f, 1, 1 ); 

#ifdef USE_KINECT
   nbPrimitives = oclKernel->addPrimitive( ptCamera );
   oclKernel->setPrimitive( nbPrimitives, 0, 100, gRoomSize-10, 320, 240, 0, 1 ); 
#endif // USE_KINECT

   std::cout << nbPrimitives+1 << " primitives" << std::endl;

   // Lamps
   nbLamps = oclKernel->addLamp();
   oclKernel->setLamp( nbLamps, 1500.0, 2000.0, -1500.0, 3.f, 1.f, 1.f, 1.f);
   std::cout << nbLamps+1 << " lamps" << std::endl;
}

void main( int argc, char* argv[] )
{
   std::cout << "--------------------------------------------------------------------------------" << std::endl;
   std::cout << "Keys:" << std::endl;
   std::cout << "  s: add sphere" << std::endl;
   std::cout << "  y: add cylinder" << std::endl;
   std::cout << "  c: add cube" << std::endl;
   std::cout << "  p: add plan (single faced)" << std::endl;
   std::cout << "  l: add lamp" << std::endl;
   std::cout << "  r: reset scene" << std::endl;
   std::cout << "Mouse:" << std::endl;
   std::cout << "  left       : Zoom in/out" << std::endl;
   std::cout << "  middle     : Rotate" << std::endl;
   std::cout << "  right      : Translate" << std::endl;
   std::cout << "  shift+left : Depth of view" << std::endl;
   std::cout << std::endl;
   std::cout << "--------------------------------------------------------------------------------" << std::endl;
   if( argc == 5 ) {
      std::cout << argv[1] << std::endl;
      sscanf_s( argv[1], "%d", &platform );
      sscanf_s( argv[2], "%d", &device );
      sscanf_s( argv[3], "%d", &window_width );
      sscanf_s( argv[4], "%d", &window_height );
   }
   else {
      std::cout << "Usage:" << std::endl;
      std::cout << "  OpenCLRaytracerTester.exe [platformId] [deviceId] [WindowWidth] [WindowHeight]" << std::endl;
      std::cout << std::endl;
      std::cout << "Example:" << std::endl;
      std::cout << "  OpenCLRaytracerTester.exe 0 1 640 480" << std::endl;
      std::cout << std::endl;
      exit(1);
   }
   initgl( argc, argv );

   // Create Scene
   createScene( platform, device );

   atexit(cleanup);
   glutMainLoop();

   // Normally unused return path
   Cleanup(EXIT_SUCCESS);
}
