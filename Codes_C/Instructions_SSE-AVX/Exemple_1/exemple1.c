/*==============================================================================*/
/* Programme 	: Exemple1.c						*/
/* Auteur 	: Daniel CHILLET						*/
/* Date 	: Novembre 2021							*/
/* 										*/
/*==============================================================================*/

#include <immintrin.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N 4

#define InitClock    struct timespec start, stop
#define ClockStart   clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start)
#define ClockEnd   clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop)
#define BILLION  1000000000L
#define ClockMesureSec "%2.9f s\n",(( stop.tv_sec - start.tv_sec )+ (stop.tv_nsec - start.tv_nsec )/(double)BILLION) 



int main() 
{ 
    float data1[4] = {1.f, 2.f, 3.f, 4.f};
    float data2[4] = {5.f, 6.f, 7.f, 8.f};
    float data3[4] ;
    float data4[4] ;

    __m128 Vec1 = _mm_loadu_ps(&data1[0]);
    __m128 Vec2 = _mm_loadu_ps(&data2[0]);
    __m128 Vec3 = _mm_set1_ps((float) 0.0);
    __m128 Vec4 = _mm_set1_ps((float) 0.0);

     printf("Vecteur1 apres declaration et initialisation\n");
   for(int i = 0; i < N; i++)
        printf("%f\n",Vec1[i]);

     printf("Vecteur2 apres declaration et initialisation\n");
   for(int i = 0; i < N; i++)
        printf("%f\n",Vec2[i]);

    Vec3 = _mm_add_ps(Vec1, Vec2);
    printf("Vecteur3 apres calcul addition\n");
   for(int i = 0; i < N; i++)
        printf("%f\n",Vec3[i]);

    Vec4 = _mm_mul_ps(Vec3, Vec2);
    printf("Vecteur4 apres calcul multiplication\n");
   for(int i = 0; i < N; i++)
        printf("%f\n",Vec4[i]);

    return 0;
}
