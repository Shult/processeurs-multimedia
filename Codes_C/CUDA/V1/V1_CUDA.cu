/*==============================================================================*/
/* Programme 	: CodeSequentiel_V2.c											*/
/* Auteur 	: Sylvain MESTRE, Antoine MAURAIS									*/
/* Date 	: Septembre 2023													*/
/* 																				*/
/*==============================================================================*/

#include <stdlib.h>
#include <stdio.h>


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
#define ClockMesureSec "%2.9f\n",(( stop.tv_sec - start.tv_sec )+ (stop.tv_nsec - start.tv_nsec )/(double)BILLION) 




#define DEBUG (0)
#define TPSCALCUL (1)

__global__ void traiterImage(int *image, int *resultat, int TailleImage, int LE_MIN, float ETALEMENT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < TailleImage){
        resultat[i] = ((image[i] - LE_MIN) * ETALEMENT);
    }
}


int main(int argc, char **argv) {
	/*========================================================================*/
	/* Declaration de variables et allocation memoire */
	/*========================================================================*/

	int i, n;

	int info ;
	
	int LE_MIN = MAX_VALEUR;
	int LE_MAX = MIN_VALEUR;
	
	float ETALEMENT = 0.0;
	
	int *image;
	int *resultat;

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

	CALLOC(image, TailleImage, int);
	CALLOC(resultat, TailleImage, int);

	if DEBUG printf("\t\t Initialisation de l'image [%d ; %d] : Ok \n", X, Y);
			
	
	x = 0;
	y = 0;
	
	lignes = 0;
	
	/*========================================================================*/
	/* Lecture du fichier pour remplir l'image source 			*/
	/*========================================================================*/
	
	while (! feof(Src)) {
		n = fscanf(Src,"%d",&P);
		image[x] = (float)P;	
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


	// Etant donnée qu'on parcours plus qu'un tableau, on a besoin de qu'une seul boucle
	for (i=0;i<TailleImage;i++) {
		LE_MIN = MIN(LE_MIN, image[i]);
		LE_MAX = MAX(LE_MAX, image[i]);
	}

	if DEBUG printf("\t Min %d ; Max %d \n\n", LE_MIN, LE_MAX);

	// Allocation de mémoire sur le GPU (CUDA)
    int *d_image, *d_resultat;
    cudaMalloc((void**)&d_image, TailleImage * sizeof(int));
    cudaMalloc((void**)&d_resultat, TailleImage * sizeof(int));

    // Copie des données de l'image source vers le GPU (CUDA)
    cudaMemcpy(d_image, image, TailleImage * sizeof(int), cudaMemcpyHostToDevice);

	/*========================================================================*/
	/* Calcul du facteur d'etalement					*/
	/*========================================================================*/
	
	if (inverse) {
		ETALEMENT = 0.2;	
	} else {
		ETALEMENT = (float)(MAX_VALEUR - MIN_VALEUR) / (float)(LE_MAX - LE_MIN);	
	}

	// Calcul des nouvelles valeurs de pixel sur le GPU (CUDA)
    int threadsPerBlock = 256; // ou toute autre valeur en fonction de votre matériel
    int blocksPerGrid = (TailleImage + threadsPerBlock - 1) / threadsPerBlock;
    traiterImage<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_resultat, TailleImage, LE_MIN, ETALEMENT);
	
	// Copie des résultats du GPU vers la mémoire hôte
    cudaMemcpy(resultat, d_resultat, TailleImage * sizeof(int), cudaMemcpyDeviceToHost);

    // Libération de la mémoire sur le GPU
    cudaFree(d_image);
    cudaFree(d_resultat);

	/*========================================================================*/
	/* Calcul de cahque nouvelle valeur de pixel							*/
	/*========================================================================*/
	
ClockStart;

	for (i = 0 ; i < TailleImage ; i++) {
		resultat[i] = ((image[i] - LE_MIN) * ETALEMENT);
	}


ClockEnd;


if TPSCALCUL printf(ClockMesureSec);

	/*========================================================================*/
	/* Sauvegarde de l'image dans le fichier resultat			*/
	/*========================================================================*/
	
	n = 0;
	for (i = 0 ; i < TailleImage ; i++) {
		fprintf(Dst,"%3d ",resultat[i]);
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
