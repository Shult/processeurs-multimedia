#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define initTimer struct timeval tv1, tv2; struct timezone tz
#define startTimer gettimeofday(&tv1, &tz)
#define stopTimer gettimeofday(&tv2, &tz)
#define tpsCalcul (tv2.tv_sec-tv1.tv_sec)*1000000L + (tv2.tv_usec-tv1.tv_usec)

int tailleVecteur ;

void add_vec_scalaire_cpu(int *vec, int *res, int a, int N) 
{
	int i ;
	for (i=0 ; i < N ; i ++) {
		res[i] = vec[i] + a;
	}
}

void main(int argc, char *argv[]) {

	if (argc != 2) {
		printf("Erreur, manque un argument\n");
		exit(0);
	}
	tailleVecteur = atoi(argv[1]);

	int vecteur[tailleVecteur];
	int resultat[tailleVecteur];
	int i ;

	initTimer;
	for (i= 0 ; i < tailleVecteur ; i++) {
		vecteur[i] = rand() % 100;
		resultat[i] = 0;
	}
	startTimer;

	add_vec_scalaire_cpu (vecteur, resultat, 10, tailleVecteur);

	stopTimer;

	printf("Vecteur %d => Temps calcul CPU %ld \n", tailleVecteur, tpsCalcul);
}

