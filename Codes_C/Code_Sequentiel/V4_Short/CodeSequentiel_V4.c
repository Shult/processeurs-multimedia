/*==============================================================================*/
/* Programme 	: CodeSequentiel_V4.c											*/
/* Auteur 	: Sylvain MESTRE, Antoine MAURAIS									*/
/* Date 	: Septembre 2023													*/
/* 																				*/
/*==============================================================================*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#define MAX_CHAINE 100
#define MAX_HOSTS 100

#define CALLOC(ptr, nr, type) 		if (!(ptr = (type *) calloc((size_t)(nr), sizeof(type)))) {		\
						printf("Erreur lors de l'allocation memoire \n") ; 		\
						exit (-1);							\
					} 

#define FOPEN(fich,fichier,sens) 	if ((fich=fopen(fichier,sens)) == NULL) { 				\
						printf("Probleme d'ouverture du fichier %s\n",fichier);		\
						exit(-1);							\
					} 
				
#define MIN(a, b) 	(a < b ? a : b)
#define MAX(a, b) 	(a > b ? a : b)

#define MAX_VALEUR 	255
#define MIN_VALEUR 	0

#define NBPOINTSPARLIGNES 15

#define false 0
#define true 1
#define boolean int


#include <time.h>

#define InitClock    struct timespec start, stop
#define ClockStart   clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start)
#define ClockEnd   clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop)
#define BILLION  1000000000L
#define ClockMesureSec "%2.9f s\n",(( stop.tv_sec - start.tv_sec )+ (stop.tv_nsec - start.tv_nsec )/(double)BILLION) 




#define DEBUG (0)
#define TPSCALCUL (1)

// =========== AJOUT ===============
#define VecteurSize 4
typedef float __m128 __attribute__((__vector_size__(16), __aligned__(16)));
// =================================

int main(int argc, char **argv) {
	/*========================================================================*/
	/* Declaration de variables et allocation memoire */
	/*========================================================================*/

	int i, n;

	// =========== AJOUT ===============
	// Décalaration des 3 vecteurs
    __m128i SSEPixelIn = _mm_set1_epi16(0);
    __m128i SSEPixelMin = _mm_set1_epi16(0);
    __m128i SSEEtalement = _mm_set1_epi16(0);
	__m128i SSEPixelOut = _mm_set1_epi16(0);
	// =================================

	int info ;
	
	int LE_MIN = MAX_VALEUR;
	int LE_MAX = MIN_VALEUR;
	
	float ETALEMENT = 0.0;
	
	short *image;
	short *resultat;

	int X, Y, x, y;
	int TailleImage;

	int lignes;
	
	int P;
	
	FILE *Src, *Dst;

	char SrcFile[MAX_CHAINE];
	char DstFile[MAX_CHAINE+4];
	
	char ligne[MAX_CHAINE];
	
	
	boolean fin ;
	boolean inverse = false;
	
	char *Chemin;
	char *CheminTache;
	

InitClock;

	/*========================================================================*/
	/* Recuperation des parametres						*/
	/*========================================================================*/


	if (argc != 2){
		printf("Syntaxe : CodeSequentiel image.pgm \n");
		exit(-1);
	}
	sscanf(argv[1],"%s", SrcFile);
	
	sprintf(DstFile,"%s.new",SrcFile);
	
	/*========================================================================*/
	/* Recuperation de l'endroit ou l'on travail				*/
	/*========================================================================*/

	CALLOC(Chemin, MAX_CHAINE, char);
	CALLOC(CheminTache, MAX_CHAINE, char);
	Chemin = getenv("PWD");
	if DEBUG printf("Repertoire de travail : %s \n\n",Chemin);

	/*========================================================================*/
	/* Ouverture des fichiers						*/
	/*========================================================================*/

	if DEBUG printf("Operations sur les fichiers\n");

	FOPEN(Src, SrcFile, "r");
	if DEBUG printf("\t Fichier source ouvert (%s) \n",SrcFile);
		
	FOPEN(Dst, DstFile, "w");
	if DEBUG printf("\t Fichier destination ouvert (%s) \n",DstFile);
	
	/*========================================================================*/
	/* On effectue la lecture du fichier source */
	/*========================================================================*/
	
	if DEBUG printf("\t Lecture entete du fichier source ");
	
	for (i = 0 ; i < 2 ; i++) {
		fgets(ligne, MAX_CHAINE, Src);	
		fprintf(Dst,"%s", ligne);
	}	

	fscanf(Src," %d %d\n",&X, &Y);
	fprintf(Dst," %d %d\n", X, Y);
	
	fgets(ligne, MAX_CHAINE, Src);	/* Lecture du 255 	*/
	fprintf(Dst,"%s", ligne);
	
	if DEBUG printf(": OK \n");
	
	/*========================================================================*/
	/* Allocation memoire pour l'image source et l'image resultat 		*/
	/*========================================================================*/
	
	TailleImage = X * Y;

	CALLOC(image, TailleImage, short);
	CALLOC(resultat, TailleImage, short);

	if DEBUG printf("\t\t Initialisation de l'image [%d ; %d] : Ok \n", X, Y);
			
	
	x = 0;
	y = 0;
	
	lignes = 0;
	
	/*========================================================================*/
	/* Lecture du fichier pour remplir l'image source 			*/
	/*========================================================================*/
	
	while (! feof(Src)) {
		n = fscanf(Src,"%d",&P);
		// =========== MODIF ===============
		image[x] = (short)P;	// Cast en short
		// =================================
		x ++;
		if (n == EOF || (x == TailleImage)) {
			break;
		}
		if (x == TailleImage) {
			x = 0 ;
		}
	}


	fclose(Src);
	if DEBUG printf("\t Lecture du fichier image : Ok \n\n");

	// =========== MODIF ===============
	// Etant donnée qu'on parcours plus qu'un tableau, on a besoin de qu'une seul boucle
	for (i=0;i<TailleImage;i++) {
		LE_MIN = MIN(LE_MIN, image[i]);
		LE_MAX = MAX(LE_MAX, image[i]);
	}
	// =================================

	if DEBUG printf("\t Min %d ; Max %d \n\n", LE_MIN, LE_MAX);


	/*========================================================================*/
	/* Calcul du facteur d'etalement					*/
	/*========================================================================*/
	
	if (inverse) {
		ETALEMENT = 0.2;	
	} else {
		ETALEMENT = (float)(MAX_VALEUR - MIN_VALEUR) / (float)(LE_MAX - LE_MIN);	
	}

	SSEEtalement = _mm_set1_epi16((short) ETALEMENT);
	SSEPixelMin = _mm_set1_epi16((short) - LE_MIN);	// On met 8 fois la valeur 

	/*========================================================================*/
	/* Calcul de chaque nouvelle valeur de pixel							*/
	/*========================================================================*/

	// ps = donnée load simple précision 
	// _mm_loadu_ps = Load 128-bits (composed of 4 packed single-precision (32-bit) floating-point elements) from memory into dst. mem_addr does not need to be aligned on any particular boundary.


ClockStart;

	// =========== MODIF ===============
	for (i = 0 ; i < TailleImage ; i = i + 8) {
		memcpy(&SSEPixelIn, &image[i], sizeof(SSEPixelIn));

		SSEPixelOut = _mm_adds_epi16(SSEPixelIn, SSEPixelMin);
		SSEPixelOut = _mm_mullo_epi16(SSEPixelOut, SSEEtalement);

		memcpy(&resultat[i], &SSEPixelOut, sizeof(SSEPixelOut));
	}
	// =================================

ClockEnd;


if TPSCALCUL printf(ClockMesureSec);

	/*========================================================================*/
	/* Sauvegarde de l'image dans le fichier resultat			*/
	/*========================================================================*/
	
	n = 0;
	for (i = 0 ; i < TailleImage ; i++) {
		fprintf(Dst,"%3d ",(int)resultat[i]);
		n++;
		if (n == NBPOINTSPARLIGNES) {
			n = 0;
			fprintf(Dst, "\n");
		}
	}
				
	fprintf(Dst,"\n");
	fclose(Dst);
	
	printf("\n");

	/*========================================================================*/
	/* Fin du programme principal	*/
	/*========================================================================*/
	
	exit(0); 
	
}
