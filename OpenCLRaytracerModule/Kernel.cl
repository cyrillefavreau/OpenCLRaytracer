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

// Max number of ray iterations
#define gNbIterations 10
// Textures
#define gTextureWidth  512
#define gTextureHeight 512
#define gTextureDepth  3

#define gVideoColor  4
#define gVideoWidth  640
#define gVideoHeight 480

#define gDepthColor       2
#define gDepthWidth       320
#define gDepthHeight      240

#define gMaxViewDistance 3000.f
#define gTextureOffset   0.f

#define gDepthOfFieldComplexity 1
#define gNbMaxShadowCollisions 3

#define NO_TEXTURE -1

#define EPSILON 1.f

// Enums
enum PrimitiveType 
{
   ptSphere     = 0,
   ptTriangle   = 1,
   ptCheckboard = 2,
   ptCamera     = 3,
   ptXYPlane    = 4,
   ptYZPlane    = 5,
   ptXZPlane    = 6,
   ptCylinder   = 7
};

typedef struct
{
   float4 color;
   float  refraction;
   int    textured;
   float  transparency;
   int    textureId;
   float4 specular;    // x: value, y: power, w: coef, z:inner illuminatino

} Material;

typedef struct 
{
   float4 center;
   //float4 rotation;
   float4 size;
   int    type;
   int    materialId;
   float  materialRatioX;
   float  materialRatioY;
} Primitive;

typedef struct 
{
   float4 center;
   float4 color;
} Lamp;

// ________________________________________________________________________________
void makeDelphiColor( 
   float4         color, 
   __global char* bitmap, 
   int            index)
{
   int mdc_index = index*3; 
   color.x = (color.x>1.f) ? 1.f : color.x;
   color.y = (color.y>1.f) ? 1.f : color.y; 
   color.z = (color.z>1.f) ? 1.f : color.z;
   bitmap[mdc_index  ] = (char)(color.z*255.f);
   bitmap[mdc_index+1] = (char)(color.y*255.f);
   bitmap[mdc_index+2] = (char)(color.x*255.f);
}

// ________________________________________________________________________________
void makeOpenGLColor( 
   float4         color, 
   __global char* bitmap, 
   int            index)
{
   int mdc_index = index*4; 
   color.x = (color.x>1.f) ? 1.f : color.x;
   color.y = (color.y>1.f) ? 1.f : color.y; 
   color.z = (color.z>1.f) ? 1.f : color.z;
   color.w = (color.w>1.f) ? 1.f : color.w;
   bitmap[mdc_index  ] = (char)(color.x*255.f); // Red
   bitmap[mdc_index+1] = (char)(color.y*255.f); // Green
   bitmap[mdc_index+2] = (char)(color.z*255.f); // Blue
   bitmap[mdc_index+3] = (char)(color.w*255.f); // Alpha
}

// ________________________________________________________________________________
#if 1
#define vectorLength( vector ) \
   (half_sqrt( (vector).x*(vector).x + (vector).y*(vector).y + (vector).z*(vector).z ))
#else
#define vectorLength( vector ) \
   fast_length(vector)
#endif

// ________________________________________________________________________________
#define normalizeVector( v ) \
   v /= vectorLength( v );

// ________________________________________________________________________________
#if 1
float dotProduct( float4 v1, float4 v2 ) 
{
   return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}
#else
#define dotProduct( v1, v2 )\
   dot(v1,v2)
#endif // 0

/*
________________________________________________________________________________
incident  : le vecteur normal inverse a la direction d'incidence de la source 
lumineuse
normal    : la normale a l'interface orientee dans le materiau ou se propage le 
rayon incident
reflected : le vecteur normal reflechi
________________________________________________________________________________
*/
#define vectorReflection( __r, __i, __n ) \
   __r = __i-2.f*dotProduct(__i,__n)*__n;

/*
________________________________________________________________________________
incident: le vecteur norm? inverse ? la direction d?incidence de la source 
lumineuse
n1      : index of refraction of original medium
n2      : index of refraction of new medium
________________________________________________________________________________
*/
void vectorRefraction( float4* refracted, float4 incident, float n1, float4 normal, float n2 )
{
	(*refracted) = incident;
	if(n1!=n2 && n2!=0.f) 
	{
		float r = n1/n2;
		float cosI = dotProduct( incident, normal );
		float cosT2 = 1.f - r*r*(1.f - cosI*cosI);
		(*refracted) = r*incident + (r*cosI-sqrt( fabs(cosT2) ))*normal;
	}
}

/*
________________________________________________________________________________
__v : Vector to rotate
__c : Center of rotations
__a : Angles
________________________________________________________________________________
*/
#define vectorRotation( __v, __c, __a ) \
{ \
   float4 __r = __v; \
   /* X axis */ \
   __r.y = __v.y*half_cos(angles.x) - __v.z*half_sin(__a.x); \
   __r.z = __v.y*half_sin(angles.x) + __v.z*half_cos(__a.x); \
   __v = __r; \
   __r = __v; \
   /* Y axis */ \
   __r.z = __v.z*half_cos(__a.y) - __v.x*half_sin(__a.y); \
   __r.x = __v.z*half_sin(__a.y) + __v.x*half_cos(__a.y); \
   __v = __r; \
}

/**
* ________________________________________________________________________________
* sphereMapping
* ________________________________________________________________________________
*/
float4 sphereMapping( 
   Primitive          primitive, 
   float4             intersection, 
   __global Material* materials, 
   __global char*     textures )
{
   float4 result = materials[primitive.materialId].color;
   int x = gTextureOffset+(intersection.x-primitive.center.x+primitive.size.x)*primitive.materialRatioX;
   int y = gTextureOffset+(intersection.y-primitive.center.y+primitive.size.y)*primitive.materialRatioY;

   x = x % gTextureWidth;
   y = y % gTextureHeight;

   if( x>=0 && x<gTextureWidth&& y>=0 && y<gTextureHeight )
   {
      int index = (materials[primitive.materialId].textureId*gTextureWidth*gTextureHeight + y*gTextureWidth+x)*gTextureDepth;
      unsigned char r = textures[index  ];
      unsigned char g = textures[index+1];
      unsigned char b = textures[index+2];
      result.x = r/256.f;
      result.y = g/256.f;
      result.z = b/256.f;
   }
   return result; 
}

/**
* ________________________________________________________________________________
* cubeMapping
* ________________________________________________________________________________
*/
float4 cubeMapping( 
   Primitive          primitive, 
   float4             intersection, 
   __global Material* materials, 
   __global char*     textures)
{
   float4 result = materials[primitive.materialId].color;
   int x = ((primitive.type == ptCheckboard) ||
            (primitive.type == ptXZPlane)    ||
            (primitive.type == ptXYPlane))  ? 
        gTextureOffset+(intersection.x-primitive.center.x+primitive.size.x)*primitive.materialRatioX:
        gTextureOffset+(intersection.z-primitive.center.z+primitive.size.x)*primitive.materialRatioX;

   int y = ((primitive.type == ptCheckboard)  ||
            (primitive.type == ptXZPlane)) ? 
        gTextureOffset+(intersection.z-primitive.center.z+primitive.size.y)*primitive.materialRatioY:
        gTextureOffset+(intersection.y-primitive.center.y+primitive.size.y)*primitive.materialRatioY;
   x = x % gTextureWidth;
   y = y % gTextureHeight;

   if( x>=0 && x<gTextureWidth&& y>=0 && y<gTextureHeight )
   {
      int index = (materials[primitive.materialId].textureId*gTextureWidth*gTextureHeight + y*gTextureWidth+x)*gTextureDepth;
      unsigned char r = textures[index];
      unsigned char g = textures[index+1];
      unsigned char b = textures[index+2];
      result.x = r/256.f;
      result.y = g/256.f;
      result.z = b/256.f;
   }
   return result;
}

/**
* ________________________________________________________________________________
* Colors
* ________________________________________________________________________________
*/
float4 objectColorAtIntersection( 
   Primitive          primitive, 
   float4             intersection,
   __global char*     video,
   __global char*     depth,
   __global Material* materials,
   __global char*     textures,
   float              timer, 
   bool               back )
{
   float4 colorAtIntersection = materials[primitive.materialId].color;
   switch( primitive.type ) 
   {
   case ptSphere:
   case ptCylinder:
      {
         colorAtIntersection = 
            ((materials[primitive.materialId].textureId != NO_TEXTURE) && (intersection.w==0.f)) ? 
            sphereMapping(primitive, intersection, materials, textures) : 
            colorAtIntersection;
         break;
      }
   case ptTriangle:
   case ptCheckboard :
      {
         if( materials[primitive.materialId].textureId != NO_TEXTURE ) 
         {
            colorAtIntersection = cubeMapping( primitive, intersection, materials, textures );
         }
         else 
         {
            int x = gMaxViewDistance + (intersection.x - primitive.center.x)/50.f;
            int z = gMaxViewDistance + (intersection.z - primitive.center.z)/50.f;
            if(x%2==0) 
            {
               if (z%2==0) 
               {
                  colorAtIntersection.x = 1.0f;
                  colorAtIntersection.y = 1.0f;
                  colorAtIntersection.z = 1.0f;
               }
            }
            else 
            {
               if (z%2!=0) 
               {
                  colorAtIntersection.x = 1.0f;
                  colorAtIntersection.y = 1.0f;
                  colorAtIntersection.z = 1.0f;
               }
            }
         }
         break;
      }
   case ptXYPlane:
   case ptYZPlane:
   case ptXZPlane:
      {
         colorAtIntersection = 
            ( materials[primitive.materialId].textureId != NO_TEXTURE ) ? 
            cubeMapping( primitive, intersection, materials, textures ) : 
            colorAtIntersection;
         break;
      }
   case ptCamera:
      {
         colorAtIntersection = materials[primitive.materialId].color;
         int x = (primitive.center.x + intersection.x)+gVideoWidth/2;
         int y = gVideoHeight/2 - (intersection.y - primitive.center.y);
         if( x>=0 && x<gVideoWidth && y>=0 && y<gVideoHeight ) 
         {
            int index = (y*gVideoWidth+x)*4;
            unsigned char r = video[index+2];
            unsigned char g = video[index+1];
            unsigned char b = video[index+0];
            colorAtIntersection.x = r/256.f;
            colorAtIntersection.y = g/256.f;
            colorAtIntersection.z = b/256.f;
         }
#if 0
         int x = 2.f*(primitive.center.x + intersection.x)+gDepthWidth/2;
         int y = gDepthHeight/2 - 2.f*(intersection.y - primitive.center.y);
         if( x>=0 && x<gDepthWidth && y>=0 && y<gDepthHeight ) {
            int index = (y*gDepthWidth+x)*2;
            char s = depth[index] & 0x07;
            //if( s != 0 ) 
            {
               char a = depth[index] & 0x1F;
               int realDepth = (depth[index+1]<<5)|(a>>3);
               float intensity = 1.f - realDepth/4000.f;
               colorAtIntersection.x = intensity;
               colorAtIntersection.y = intensity;
               colorAtIntersection.z = intensity;
            }
         }
#endif // 0
         break;
      }
   }
   return colorAtIntersection;
}

/**
________________________________________________________________________________
Lamp Intersection
Lamp         : Object on which we want to find the intersection
origin       : Origin of the ray
orientation  : Orientation of the ray
intersection : Resulting intersection
invert       : true if we want to check the front of the sphere, false for the 
back
returns true if there is an intersection, false otherwise
________________________________________________________________________________
*/
bool lampIntersection( 
   Lamp    lamp, 
   float4  origin, 
   float4  ray, 
   float4  O_C,
   float4* intersection)
{
   float si_A = 2.f*(ray.x*ray.x + ray.y*ray.y + ray.z*ray.z);
   if ( si_A == 0.f ) return false;

   bool  si_b1 = false; 
   float si_B = 2.f*(O_C.x*ray.x + O_C.y*ray.y + O_C.z*ray.z);
   float si_C = O_C.x*O_C.x+O_C.y*O_C.y+O_C.z*O_C.z-lamp.center.w*lamp.center.w;
   float si_radius = si_B*si_B-2.f*si_A*si_C;
   float si_t1 = (-si_B-half_sqrt(si_radius))/si_A;

   if( si_t1>0.f ) 
   {
      *intersection = origin+si_t1*ray;
      si_b1 = true;
   }
   return si_b1;
}


/**
________________________________________________________________________________
Sphere Intersection
primitive    : Object on which we want to find the intersection
origin       : Origin of the ray
orientation  : Orientation of the ray
intersection : Resulting intersection
invert       : true if we want to check the front of the sphere, false for the 
back
returns true if there is an intersection, false otherwise
________________________________________________________________________________
*/
bool sphereIntersection( 
   Primitive          sphere, 
   float4             origin, 
   float4             ray, 
   float              timer,
   float4*            intersection,
   float4*            normal,
   bool               computingShadows,
   float*             shadowIntensity,
   __global char*     video,
   __global char*     depth,
   __global Material* materials,
   __global char*     textures,
   float              transparentColor,
   bool*              back
   ) 
{
	// solve the equation sphere-ray to find the intersections
	float4 O_C = origin-sphere.center;
	float4 dir = ray;
	normalizeVector( dir );

	float a = 2.f*dotProduct(dir,dir);
	float b = 2.f*dotProduct(O_C,dir);
	float c = dotProduct(O_C,O_C) - (sphere.size.x*sphere.size.x);
	float d = b*b-2.f*a*c;

	if( d<=0.f || a == 0.f) return false;
	float r = sqrt(d);
	float t1 = (-b-r)/a;
	float t2 = (-b+r)/a;

	if( t1<=EPSILON && t2<=EPSILON ) return false; // both intersections are behind the ray origin
	*back = (t1<=EPSILON || t2<=EPSILON); // If only one intersection (t>0) then we are inside the sphere and the intersection is at the back of the sphere

	float t=0.f;
	if( t1<=EPSILON ) 
		t = t2;
	else 
		if( t2<=EPSILON )
			t = t1;
		else
			t=(t1<t2) ? t1 : t2;

	if( t<EPSILON ) return false; // Too close to intersection
	*intersection = origin+t*dir;

	// Compute normal vector
	(*normal) = *intersection-sphere.center;
	(*normal).w = 0.f;
	(*normal) *= (back) ? -1.f : 1.f;
	normalizeVector(*normal);

	return true;
}

/**
________________________________________________________________________________
Cylinder Intersection
primitive    : Object on which we want to find the intersection
origin       : Origin of the ray
orientation  : Orientation of the ray
intersection : Resulting intersection
invert       : true if we want to check the front of the sphere, false for the 
back
returns true if there is an intersection, false otherwise
________________________________________________________________________________
*/
bool cylinderIntersection( 
   Primitive          cylinder, 
   float4             origin, 
   float4             ray, 
   float              timer,
   float4*            intersection,
   float4*            normal,
   bool               computingShadows,
   float*             shadowIntensity,
   __global char*     video,
   __global char*     depth,
   __global Material* materials,
   __global char*     textures,
   float              transparentColor
   ) 
{
   // solve the equation sphere-ray to find the intersections
   bool result = false;
   //bool reverseNormal = false;

   // Top
   if(!result && ray.y<0.f && origin.y>(cylinder.center.y+cylinder.size.y)) 
   {
      (*intersection).y = cylinder.center.y+cylinder.size.y;
      float y = origin.y-cylinder.center.y-cylinder.size.y;
      (*intersection).x = origin.x+y*ray.x/-ray.y;
      (*intersection).z = origin.z+y*ray.z/-ray.y;
      (*intersection).w = 1.f; // 1 for top, -1 for bottom

      float4 v=(*intersection)-cylinder.center;
      v.y = 0.f;
      result = (vectorLength(v)<cylinder.size.x);

      (*normal).x =  0.f;
      (*normal).y =  1.f;
      (*normal).z =  0.f;
   }

   // Bottom
   if( !result && ray.y>0.f && origin.y<(cylinder.center.y - cylinder.size.y) ) 
   {
      (*intersection).y = cylinder.center.y - cylinder.size.y;
      float y = origin.y - cylinder.center.y + cylinder.size.y;
      (*intersection).x = origin.x+y*ray.x/-ray.y;
      (*intersection).z = origin.z+y*ray.z/-ray.y;
      (*intersection).w = -1.f; // 1 for top, -1 for bottom

      float4 v=(*intersection)-cylinder.center;
      v.y = 0.f;
      result = (vectorLength(v)<cylinder.size.x);

      (*normal).x =  0.f;
      (*normal).y = -1.f;
      (*normal).z =  0.f;
   }

   if( !result ) 
   {
      float4 O_C = origin - cylinder.center;
      O_C.y = 0.f;
      if(( dotProduct( O_C, ray ) > 0.f ) && (vectorLength(O_C) > cylinder.center.w)) return false;

      float a = 2.f * ( ray.x*ray.x + ray.z*ray.z );
      float b = 2.f*((origin.x-cylinder.center.x)*ray.x + (origin.z-cylinder.center.z)*ray.z);
      float c = O_C.x*O_C.x + O_C.z*O_C.z - cylinder.center.w*cylinder.center.w;

      float r = half_sqrt(b*b-2.f*a*c);

      // Cylinder
      if ( r < 0.f ) return false;
   
      a = ( a==0.f ) ? 0.0001f : a;
      float t1 = (-b-r)/a;
      float t2 = (-b+r)/a;
      float ta = (t1<t2) ? t1 : t2;
      float tb = (t2<t1) ? t1 : t2;
   
      if( ta >  0.001f ) 
      {
         *intersection = origin+ta*ray;
         (*intersection).w = 0.f;

         result = ( fabs((*intersection).y - cylinder.center.y) <= cylinder.size.y );
         if( result && materials[cylinder.materialId].transparency != 0.f ) 
         {
            float4 color = objectColorAtIntersection( cylinder, *intersection, video, depth, materials, textures, timer, false );
            result = 
               ( fabs((*intersection).y - cylinder.center.y) <= cylinder.size.y ) &&
               ( (color.x+color.y+color.z) >= transparentColor ); 
         }
      }

      if( !result && tb > 0.001f ) 
      {
         *intersection = origin+tb*ray;
         (*intersection).w = 0.f;

         result = ( fabs((*intersection).y - cylinder.center.y) <= cylinder.size.y );
         //reverseNormal = true;
         if( result && materials[cylinder.materialId].transparency != 0.f ) 
         {
            float4 color = objectColorAtIntersection( cylinder, *intersection, video, depth, materials, textures, timer, false );
            result = 
               ( fabs((*intersection).y - cylinder.center.y) <= cylinder.size.y ) &&
               ( (color.x+color.y+color.z) >= transparentColor ); 
         }
      }

      if( result )
      {
         (*normal) = (*intersection)-cylinder.center;
         (*normal).y = 0.f;
      }
   }

   // Normal to surface
   if( result && !computingShadows ) 
   {
      if( materials[cylinder.materialId].textured ) 
      {
         float4 newCenter;
         newCenter.x = cylinder.center.x + 5.f*half_cos(timer*0.58f+(*intersection).x);
         newCenter.y = cylinder.center.y + 5.f*half_sin(timer*0.85f+(*intersection).y) + (*intersection).y;
         newCenter.z = cylinder.center.z + 5.f*half_sin(half_cos(timer*1.24f+(*intersection).z));
         *normal = *intersection-newCenter;
      }
      //*normal *= reverseNormal ? -1.f : 1.f; 
      normalizeVector( *normal );
   }

#if 0
   if( result && computingShadows ) 
   {
      float4 normal = normalToSurface( cylinder, *intersection, depth, materials, timer ); // Normal is computed twice!!!
      normalizeVector(ray );
      normalizeVector(normal);
      *shadowIntensity = 5.f*fabs(dotProduct(-ray ,normal));
      *shadowIntensity = (*shadowIntensity>1.f) ? 1.f : *shadowIntensity;
   } 
#else
   *shadowIntensity = 1.f;
#endif // 0
   return result;
}

/**
________________________________________________________________________________
Checkboard Intersection
primitive    : Object on which we want to find the intersection
origin       : Origin of the ray
orientation  : Orientation of the ray
intersection : Resulting intersection
invert       : true if we want to check the front of the sphere, false for the 
back
returns true if there is an intersection, false otherwise
________________________________________________________________________________
*/
bool planeIntersection( 
   Primitive          primitive, 
   float4             origin, 
   float4             ray, 
   bool               reverse,
   float*             shadowIntensity,
   __global char*     depth,
   __global Material* materials,
   __global char*     textures,
   float4*            intersection,
   float4*            normal,
   float              transparentColor)
{ 
   //vectorRotation( &origin, primitive.center, primitive.rotation );
   //vectorRotation( &ray,    primitive.center, primitive.rotation );
   float reverted = reverse ? -1.f : 1.f;
   bool collision = false;
   switch( primitive.type ) 
   {
      case ptCheckboard:
         {
            (*intersection).y = primitive.center.y;
            float y = origin.y-primitive.center.y;
            if( reverted*ray.y<0.f && reverted*origin.y>reverted*primitive.center.y) 
            {
               (*intersection).x = origin.x+y*ray.x/-ray.y;
               (*intersection).z = origin.z+y*ray.z/-ray.y;
               collision = 
                  fabs((*intersection).x - primitive.center.x) < primitive.size.x &&
                  fabs((*intersection).z - primitive.center.z) < primitive.size.y;
               (*normal).x =  0.f;
               (*normal).y =  1.f;
               (*normal).z =  0.f;
            }
            break;
         }
      case ptXZPlane:
         {
            float y = origin.y-primitive.center.y;
            if( reverted*ray.y<0.f && reverted*origin.y>reverted*primitive.center.y) 
            {
               (*normal).x =  0.f;
               (*normal).y =  1.f;
               (*normal).z =  0.f;
               (*intersection).y = primitive.center.y;
               (*intersection).x = origin.x+y*ray.x/-ray.y;
               (*intersection).z = origin.z+y*ray.z/-ray.y;
               collision = 
                  fabs((*intersection).x - primitive.center.x) < primitive.size.x &&
                  fabs((*intersection).z - primitive.center.z) < primitive.size.y;
            }
            if( !collision && reverted*ray.y>0.f && reverted*origin.y<reverted*primitive.center.y) 
            {
               (*intersection).x = origin.x+y*ray.x/-ray.y;
               (*intersection).z = origin.z+y*ray.z/-ray.y;
               collision = 
                  fabs((*intersection).x - primitive.center.x) < primitive.size.x &&
                  fabs((*intersection).z - primitive.center.z) < primitive.size.y;
               (*normal).x =  0.f;
               (*normal).y =  -1.f;
               (*normal).z =  0.f;
            }
            break;
         }
      case ptYZPlane:
         {
            float x = origin.x-primitive.center.x;
            if( reverted*ray.x<0.f && reverted*origin.x>reverted*primitive.center.x ) 
            {
               (*intersection).x = primitive.center.x;
               (*intersection).y = origin.y+x*ray.y/-ray.x;
               (*intersection).z = origin.z+x*ray.z/-ray.x;
               collision = 
                  fabs((*intersection).y - primitive.center.y) < primitive.size.y &&
                  fabs((*intersection).z - primitive.center.z) < primitive.size.x;
               (*normal).x =  1.f;
               (*normal).y =  0.f;
               (*normal).z =  0.f;
            }
            if( !collision && reverted*ray.x>0.f && reverted*origin.x<reverted*primitive.center.x ) 
            {
               (*intersection).x = primitive.center.x;
               (*intersection).y = origin.y+x*ray.y/-ray.x;
               (*intersection).z = origin.z+x*ray.z/-ray.x;
               collision = 
                  fabs((*intersection).y - primitive.center.y) < primitive.size.y &&
                  fabs((*intersection).z - primitive.center.z) < primitive.size.x;
               (*normal).x = -1.f;
               (*normal).y =  0.f;
               (*normal).z =  0.f;
            }
            break;
         }
      case ptXYPlane:
         {
            float z = origin.z-primitive.center.z;
            if( reverted*ray.z<0.f && reverted*origin.z>reverted*primitive.center.z) 
            {
               (*intersection).z = primitive.center.z;
               (*intersection).x = origin.x+z*ray.x/-ray.z;
               (*intersection).y = origin.y+z*ray.y/-ray.z;
               collision = 
                  fabs((*intersection).x - primitive.center.x) < primitive.size.x &&
                  fabs((*intersection).y - primitive.center.y) < primitive.size.y;
               (*normal).x =  0.f;
               (*normal).y =  0.f;
               (*normal).z =  1.f;
            }
            if( !collision && reverted*ray.z>0.f && reverted*origin.z<reverted*primitive.center.z )
            {
               (*intersection).z = primitive.center.z;
               (*intersection).x = origin.x+z*ray.x/-ray.z;
               (*intersection).y = origin.y+z*ray.y/-ray.z;
               collision = 
                  fabs((*intersection).x - primitive.center.x) < primitive.size.x &&
                  fabs((*intersection).y - primitive.center.y) < primitive.size.y;
               (*normal).x =  0.f;
               (*normal).y =  0.f;
               (*normal).z = -1.f;
            }
            break;
         }
      case ptCamera:
         {
            if( reverted*ray.z>0.f && reverted*origin.z<reverted*primitive.center.z )
            {
               (*intersection).z = primitive.center.z;
               float z = origin.z-primitive.center.z;
               (*intersection).x = origin.x+z*ray.x/-ray.z;
               (*intersection).y = origin.y+z*ray.y/-ray.z;
               collision =
                  fabs((*intersection).x - primitive.center.x) < primitive.size.x &&
                  fabs((*intersection).y - primitive.center.y) < primitive.size.y;
               (*normal).x =  0.f;
               (*normal).y =  0.f;
               (*normal).z = -1.f;
            }
            break;
         }
      }

   if( collision ) 
   {
      if( /*materials[primitive.materialId].color.w != 0.f &&*/
         materials[primitive.materialId].transparency != 0.f && 
         materials[primitive.materialId].textureId!=NO_TEXTURE ) 
      {
         float4 color = cubeMapping(primitive, *intersection, materials, textures );
         *shadowIntensity = (color.x+color.y+color.z)/3.f;
         collision = ( *shadowIntensity >= transparentColor );
      }
      else 
      {
         *shadowIntensity = 1.f;
      }
   }
   //vectorRotation( intersection, primitive.center, -primitive.rotation );
   return collision;
}

/*
Shadows computation
We do not consider the object from which the ray is launched...
This object cannot shadow itself !

We now have to find the intersection between the considered object and the ray which origin is the considered 3D float4
and which direction is defined by the light source center.

* Lamp                     Ray = Origin -> Light Source Center
\
\##
#### object
##
\
\  Origin
--------O-------
*/
float shadow( 
   __global Primitive* primitives, 
   int                 nbPrimitives, 
   float4              lampCenter, 
   float4              origin, 
   int                 objectId, 
   float               timer,
   __global char*      video,
   __global char*      depth,
   __global Material*  materials, 
   __global char*      textures,
   float               transparentColor)
{
   return 0.f; // TO REMOVE!!!!


   float result = 0.f;
   float4 O_L = lampCenter - origin;
   int cptPrimitives = 0;
   int collision = 0;
   while( result<1.f && (collision<gNbMaxShadowCollisions) && (cptPrimitives<nbPrimitives) ) 
   {
      float4 intersection = 0;
      float4 normal = 0;
      float shadowIntensity = 0.f;
      bool hit = false;
      bool back;

      switch(primitives[cptPrimitives].type)
      {
      case ptSphere  : hit = sphereIntersection( primitives[cptPrimitives], origin, O_L, timer, &intersection, &normal, true, &shadowIntensity, video, depth, materials, textures, transparentColor, &back ); break;
      case ptCylinder: hit = cylinderIntersection( primitives[cptPrimitives], origin, O_L, timer, &intersection, &normal, true, &shadowIntensity, video, depth, materials, textures, transparentColor ); break;
      default        : 
         hit = planeIntersection( primitives[cptPrimitives], origin, O_L, true, &shadowIntensity, depth, materials, textures, &intersection, &normal, transparentColor ); 
         if( hit ) 
         {
            float4 O_I = intersection-origin;
            hit = ( vectorLength(O_I)<vectorLength(O_L) );
         }
         break;
      }

      if( hit ) 
      {
         collision++;
         if( collision == gNbMaxShadowCollisions ) 
         {
            result = 1.f;
         }
         else
         {
            shadowIntensity *= 
               (materials[primitives[cptPrimitives].materialId].transparency != 0.f) ?
               1.f - materials[primitives[cptPrimitives].materialId].transparency :  // Shadow intensity of a transparent object
               1.f;

            if( primitives[cptPrimitives].type == ptSphere || primitives[cptPrimitives].type == ptCylinder )
            {
               float4 O_I = intersection-origin;
               // Shadow exists only if object is between origin and lamp
               shadowIntensity = (vectorLength(O_I) < vectorLength(O_L)) ? shadowIntensity : 0.f;
            }
            result += shadowIntensity;
         }
      }
      cptPrimitives++; 
   }

   return (result>1.f) ? 1.f : result;
}


/*
* colorFromObject 
*/
float4 colorFromObject(
   __global Primitive* primitives, 
   int                 nbPrimitives, 
   __global Lamp*      lamps, 
   int                 NbLamps, 
   __global char*      video,
   __global char*      depth,
   __global Material*  materials,
   __global char*      textures,
   float4              origin,
   float4              normal, 
   int                 objectId, 
   float4              intersection, 
   float               timer,
   float4*             refractionFromColor,
   float*              shadowIntensity,
   float*              totalBlinn,
   float               transparentColor)
{
   float4 color = 0;
   float4 lampsColor = 0;

   // Lamp Impact
   float lambert = 0.f;
   float totalIntensity = 0.f;
   *totalBlinn = 0.f;

   for( int cptLamps=0; cptLamps<NbLamps; cptLamps++ ) 
   {
      *shadowIntensity = shadow( primitives, nbPrimitives, lamps[cptLamps].center, intersection, objectId, timer, video, depth, materials, textures, transparentColor );

      // Lighted object, not in the shades
      if( (*shadowIntensity) != 1.0f )
      {
         float4 lightRay = lamps[cptLamps].center - intersection;
         lampsColor += lamps[cptLamps].color;

         // Lambert
         normalizeVector(lightRay);
         lambert = dotProduct(lightRay, normal);
         lambert = (lambert<0.f) ? 0.f : lambert;
         lambert *= (materials[primitives[objectId].materialId].refraction == 0.f) ? lamps[cptLamps].color.w : 1.f;
         lambert *= (1.f-*shadowIntensity);

         totalIntensity += lambert; // + material.specular.z; // Lambert + inner illumination

         // --------------------------------------------------------------------------------
         // Blinn - Phong
         // --------------------------------------------------------------------------------
         float4 viewRay = intersection - origin;
         normalizeVector(viewRay);

         float4 blinnDir = lightRay - viewRay;
         float temp = half_sqrt(dotProduct(blinnDir,blinnDir));
         if (temp != 0.f ) 
         {
            // Specular reflection
            blinnDir = (1.f / temp) * blinnDir;

            float blinnTerm = dotProduct(blinnDir,normal);
            blinnTerm = ( blinnTerm < 0.f) ? 0.f : blinnTerm;

            blinnTerm = 
               materials[primitives[objectId].materialId].specular.x * 
               pow(blinnTerm , materials[primitives[objectId].materialId].specular.y) * 
               materials[primitives[objectId].materialId].specular.w;

            *totalBlinn += lamps[cptLamps].color.w * blinnTerm;
         }
      }
   }

   // Final color
   float4 intersectionColor = objectColorAtIntersection( primitives[objectId], intersection, video, depth, materials, textures, timer, false );

   color   = intersectionColor*lampsColor;
   color.w = totalIntensity;
   
   *refractionFromColor = intersectionColor; // Refraction depending on color;
   *totalBlinn = (*totalBlinn>1.f) ? 1.f : *totalBlinn;

   return color;
}

/**
________________________________________________________________________________
Plan Intersection
returns true if there is an intersection, false otherwise
________________________________________________________________________________
*/
bool planIntersection( 
   Primitive          plan, 
   float4             origin, 
   float4             ray, 
   __global Material* materials,
   __global char*     depth,
   float              timer, 
   float4*            intersection )
{
   bool collision  = false;
#if 0
   float4 normal = normalToSurface( plan, plan.center, depth, materials, timer  );
   normalizeVector(normal);
   float B = dotProduct( normal, ray );
   if(B < 0.f) 
   {
      origin = origin + plan.center;
      float t = -dotProduct( normal, origin )/B;
      *intersection = origin + t*ray;
      collision  = true;
   } 
#endif // 0
   return collision;
}

/**
* ________________________________________________________________________________
* Intersections with Objects
* ________________________________________________________________________________
*/
bool intersectionWithLamps( 
   __global Lamp*      lamps, 
   int                 nbLamps, 
   float4              origin, 
   float4              target, 
   float4*             lampColor)
{
   bool intersections = false; 

   for( int cptLamps = 0; cptLamps<nbLamps && !intersections; cptLamps++ ) 
   {
      float4 O_C = origin - lamps[cptLamps].center; 
      float4 ray = target - origin;
      float4 intersection;
      intersections = lampIntersection( lamps[cptLamps], origin, ray, O_C, &intersection );
      if( intersections ) 
      {
         float4 I_C = intersection - lamps[cptLamps].center;
         normalizeVector( O_C );
         normalizeVector( I_C );
         float d = dotProduct( O_C, I_C ) * lamps[cptLamps].color.w;
         d = (d<0.f) ? 0.f : d;
         (*lampColor) = lamps[cptLamps].color*d;
      }
   }
   return intersections;
}

/**
* ________________________________________________________________________________
* Intersections with Objects
* ________________________________________________________________________________
*/
bool intersectionWithPrimitives( 
   __global Primitive* primitives, 
   int                 nbPrimitives, 
   float4              origin, 
   float4              target, 
   float               timer, 
   int*                closestPrimitive, 
   float4*             closestIntersection,
   float4*             closestNormal,
   __global char*      video,
   __global char*      depth,
   __global Material*  materials,
   __global char*      textures,
   float               transparentColor,
   bool* back)
{
   bool intersections = false; 
   float minDistance  = gMaxViewDistance; 
   float4 ray = target - origin; 
   float4 intersection = 0;
   float4 normal = 0;

   for( int cptObjects = 0; cptObjects<nbPrimitives; cptObjects++ )
   { 
      bool i = false; 
      float shadowIntensity;

      switch( primitives[cptObjects].type )
      {
      case ptSphere  : i = sphereIntersection( primitives[cptObjects], origin, ray, timer, &intersection, &normal, false, &shadowIntensity, video, depth, materials, textures,transparentColor, back ); break;
      case ptCylinder: i = cylinderIntersection( primitives[cptObjects], origin, ray, timer, &intersection, &normal, false, &shadowIntensity, video, depth, materials, textures, transparentColor); break;
      case ptTriangle: i = planIntersection( primitives[cptObjects], origin, ray, materials, depth, timer, &intersection ); break;
      default        : i = planeIntersection( primitives[cptObjects], origin, ray, false, &shadowIntensity, depth, materials, textures, &intersection, &normal, transparentColor); break;
      }

      if( i ) 
      {
         float distance = vectorLength( origin - intersection );

         if(distance>0.01f && distance<minDistance) 
         {
            minDistance          = distance;
            *closestPrimitive    = cptObjects;
            *closestIntersection = intersection;
            *closestNormal       = normal;
            intersections = true;
         } 
      }
   }
   return intersections;
}

/**
*  ------------------------------------------------------------------------------ 
* Ray Intersections
*  ============================================================================== 
*  Calculate the reflected vector                   
*                                                  
*                  ^ Normal to object surface (N)  
* Reflection (O_R)  |                              
*                 \ |  Eye (O_E)                    
*                  \| /                             
*   ----------------O--------------- Object surface 
*          closestIntersection                      
*                                                   
*  ============================================================================== 
*  colours                                                                                    
*  ------------------------------------------------------------------------------ 
*  We now have to know the colour of this intersection                                        
*  Color_from_object will compute the amount of light received by the
*  intersection float4 and  will also compute the shadows. 
*  The resulted color is stored in result.                     
*  The first parameter is the closest object to the intersection (following 
*  the ray). It can  be considered as a light source if its inner light rate 
*  is > 0.                            
*  ------------------------------------------------------------------------------ 
*/
float4 launchRay( 
   __global Primitive* primitives, 
   int                 nbPrimitives, 
   __global Lamp*      lamps, 
   int                 nbLamps, 
   float4              origin, 
   float4              target, 
   float               timer,
   __global Material*  materials,
   __global char*      textures,
   __global char*      video,
   __global char*      depth,
   float               transparentColor,
   float4*             intersection)
{
   float4 intersectionColor = 0;
   int    closestPrimitive;
   float4 closestIntersection = 0;
   bool   carryon           = true;
   float4 rayOrigin         = origin;
   float4 rayTarget         = target;
   float  initialRefraction = 1.0f;
   int    iteration         = 0;
   float4 O_R;
   float4 O_E;
   float4 recursiveColor[10];
   float4 recursiveRatio[10];

   for( int i=0; i<10; i++ ) 
   {
      recursiveColor[i] = 0.f;
      recursiveRatio[i] = 0.f;
   }

   // Variable declarations
   float  shadowIntensity = 0.f;
   float4 refractionFromColor;
   float4 reflectedTarget;
   float4 normal = 0;
   float  blinn = 0.f;
   int inters=0;
   bool back;

   while( iteration<gNbIterations && carryon ) 
   {

      // Compute intesection with lamps
      carryon = !intersectionWithLamps( lamps, nbLamps, rayOrigin, rayTarget, &intersectionColor);

      // If no intersection with lamps detected. Now compute intersection with Primitives
      if( carryon ) 
      {
         carryon = intersectionWithPrimitives(
            primitives, nbPrimitives,
            rayOrigin, rayTarget,
            timer, 
            &closestPrimitive, &closestIntersection, &normal,
            video, depth, materials, textures, transparentColor,
            &back);
      }

      if( carryon ) 
      {
         inters += (back) ? -1 : 1;

         // Get object color
         recursiveColor[iteration] = colorFromObject( 
            primitives, nbPrimitives, lamps, nbLamps, 
            video, depth, materials, textures, 
            origin, normal, closestPrimitive, closestIntersection, 
            timer, &refractionFromColor, &shadowIntensity, &blinn, transparentColor );

         recursiveRatio[iteration].y = blinn;

         if( materials[primitives[closestPrimitive].materialId].transparency != 0.f ) 
         {
            // ----------
            // Refraction
            // ----------
            // Replace the normal using the intersection color
            // r,g,b become x,y,z... What the fuck!!
            if( materials[primitives[closestPrimitive].materialId].textureId != NO_TEXTURE) 
            {
               refractionFromColor -= 0.5f;
               normal *= refractionFromColor;
            }
             
            O_E = rayOrigin - closestIntersection;
            normalizeVector(O_E);
            float refraction = materials[primitives[closestPrimitive].materialId].refraction;
            refraction = (refraction == initialRefraction) ? 1.0f : refraction;
            vectorRefraction( &O_R, O_E, refraction, normal, initialRefraction );
            reflectedTarget = closestIntersection - O_R;
               
            initialRefraction = refraction;

            recursiveRatio[iteration].x = materials[primitives[closestPrimitive].materialId].transparency;
            recursiveRatio[iteration].z = 1.f;
         }
         else 
         {
            // ----------
            // Reflection
            // ----------
            if( materials[primitives[closestPrimitive].materialId].color.w != 0.f ) 
            {
               O_E = rayOrigin - closestIntersection;
               vectorReflection( O_R, O_E, normal );
               reflectedTarget = closestIntersection - O_R;

               recursiveRatio[iteration].x = materials[primitives[closestPrimitive].materialId].color.w;
               //carryon &= (shadowIntensity!=1.f);
            }
            else 
            {
               carryon = false;
            }
         }
         rayOrigin = closestIntersection; 
         rayTarget = reflectedTarget;
         iteration++; 
      }
   }

   for( int i=iteration-1; i>=0; --i ) 
   {
      float w = recursiveColor[i].w;
      if( recursiveRatio[i].z == 1.f ) {
         w = recursiveColor[i].x + w*(1.f-recursiveColor[i].x);
      }
      recursiveColor[i] = (recursiveColor[i+1]*w*recursiveRatio[i].x + recursiveColor[i]*w*(1.f-recursiveRatio[i].x));
   }
   intersectionColor = recursiveColor[0];
   
   // Specular reflection
   intersectionColor += recursiveRatio[0].y;

   intersectionColor.x = (intersectionColor.x>1.f) ? 1.f : intersectionColor.x;
   intersectionColor.y = (intersectionColor.y>1.f) ? 1.f : intersectionColor.y;
   intersectionColor.z = (intersectionColor.z>1.f) ? 1.f : intersectionColor.z;
   *intersection = closestIntersection;

#if 0
   // --------------------------------------------------
   // Attenation effect (Fog)
   // --------------------------------------------------
   float4 O_I = closestIntersection - origin;
   float len = 1.f-(vectorLength(O_I)/gMaxViewDistance);
   len = (len>0.f) ? len : 0.f; 
   intersectionColor *= len;
#endif // 0

   return intersectionColor;
}


/**
* ________________________________________________________________________________
* Main Kernel!!!
* ________________________________________________________________________________
*/
__kernel void render_kernel( 
   float4               origin,
   float4               target,
   float4               angles,
   int                  width, 
   int                  height, 
   __global Primitive*  primitives, 
   __global Lamp*       lamps, 
   __global Material*   materials, 
   int                  nbPrimitives,
   int                  nbLamps,
   int                  nbMaterials,
   __global char*       bitmap,
   __global char*       video,
   __global char*       depth,
   __global char*       textures,
   float                timer,
   int                  draft,
   float                transparentColor)
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   int index = y*width+x;

   target.x = target.x + (float)(x - (width/2));
   target.y = target.y + (float)(y - (height/2));

   float4 rotationCenter = 0;

   vectorRotation( origin, rotationCenter, angles );
   vectorRotation( target, rotationCenter, angles );

   float4 intersection;
   float4 color = launchRay( 
      primitives, nbPrimitives, 
      lamps, nbLamps, 
      origin, target, timer, 
      materials, textures,
      video, depth, transparentColor,
      &intersection);

   color.w = gMaxViewDistance/intersection.z;
   for( int j=0; j<draft; j++ ) 
   {
      makeOpenGLColor( color, bitmap, index+j ); 
   }
}
