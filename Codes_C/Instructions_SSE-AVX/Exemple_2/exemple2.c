
/*==============================================================================*/
/* Programme 	: Exemple2.c						*/
/* Auteur 	: Daniel CHILLET						*/
/* Date 	: Novembre 2021							*/
/* 										*/
/*==============================================================================*/

#include <immintrin.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define SSESIZE 4

#define VecteurSize 20 

#define InitClock    struct timespec start, stop
#define ClockStart   clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start)
#define ClockEnd   clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop)
#define BILLION  1000000000L
#define ClockMesureSec "%2.9f s\n",(( stop.tv_sec - start.tv_sec )+ (stop.tv_nsec - start.tv_nsec )/(double)BILLION) 



int main() 
{ 
	int i;

    float data1[VecteurSize] ;
    float data2[VecteurSize] ;
    float data3[VecteurSize] ;
    float data4[VecteurSize] ;

    __m128 Vec1 = _mm_set1_ps((float) 0.0);
    __m128 Vec2 = _mm_set1_ps((float) 0.0);
    __m128 Vec3 = _mm_set1_ps((float) 0.0);
    __m128 Vec4 = _mm_set1_ps((float) 0.0);

   for ( i = 0 ; i < VecteurSize ; i++) {
    		data1[i] = i;
    		data2[i] = VecteurSize - i;
    }

    printf("Vecteur1 apres declaration et initialisation\n");
   for( i = 0; i < SSESIZE; i++)
        printf("%f\n",data1[i]);

     printf("Vecteur2 apres declaration et initialisation\n");
   for( i = 0; i < SSESIZE; i++)
        printf("%f\n",data2[i]);

    for (i= 0 ; i < VecteurSize ; i = i + SSESIZE) {
	    Vec1 = _mm_loadu_ps(&data1[i]);
	    Vec2 = _mm_loadu_ps(&data2[i]);
	    Vec3 = _mm_add_ps(Vec1, Vec2);
	    _mm_store_ps(&data3[i], Vec3);
	 }

    printf("Vecteur3 apres calcul addition\n");
   for( i = 0; i < VecteurSize; i++)
        printf("%f\n",data3[i]);

    for (i= 0 ; i < VecteurSize ; i = i + SSESIZE) {
	    Vec1 = _mm_loadu_ps(&data1[i]);
	    Vec3 = _mm_loadu_ps(&data3[i]);
	    Vec4 = _mm_mul_ps(Vec1, Vec3);
	    _mm_store_ps(&data4[i], Vec4);
	 }

    printf("Vecteur4 apres calcul multiplication\n");
   for( i = 0; i < VecteurSize; i++)
        printf("%f\n",data4[i]);

    return 0;
}


