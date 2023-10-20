#include <stdio.h>
#include <sys/time.h>

#define initTimer struct timeval tv1, tv2; struct timezone tz
#define startTimer gettimeofday(&tv1, &tz)
#define stopTimer gettimeofday(&tv2, &tz)
#define tpsCalcul (tv2.tv_sec-tv1.tv_sec)*1000000L + (tv2.tv_usec-tv1.tv_usec)


#define MAX_DIM_GRID 65535
#define MAX_DIM_BLOCK 1024


long tailleVecteur ;

/* KERNEL CUDA */

__global__ void add_vec_scalaire_gpu(int *vec, int *res, int a, long N) {
	long i = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
	if (i < N) {
		res[i] = vec[i] + a;
	}
}

void add_vec_scalaire_cpu(int *vec, int *res, int a, long N) 
{
	int i ;
	for (i=0 ; i < N ; i ++) {
		res[i] = vec[i] + a;
	}
}

int main(int argc, char *argv[]) {
	int alpha = 10;
	if (argc < 2) {
		printf("Erreur, manque un argument\n");
		exit(0);
	}
	tailleVecteur = atol(argv[1]);
	long blocksize = 1;	
	if (argc ==3) {
		blocksize = atoi(argv[2]);
	}

	int *vecteur;
	int *resultat;
	int *cudaVec;
	int *cudaRes;

	initTimer;

	long size = sizeof(int)*tailleVecteur;

	vecteur = (int *)malloc(size);
	resultat = (int *)malloc(size);

	if (vecteur == NULL) {
		printf("Allocation memoire qui pose probleme (vecteur) \n");
	}
	if (resultat == NULL) {
		printf("Allocation memoire qui pose probleme (resultat) \n");
	}

	long i ;
	for (i= 0 ; i < tailleVecteur ; i++) {
		vecteur[i] = rand() % 100;
		resultat[i] = 0;
	}

/*	cudaSetDevice(1);	*/

	if (cudaMalloc((void **)&cudaVec, size) == cudaErrorMemoryAllocation) {
		printf("Allocation memoire qui pose probleme (cudaVec) \n");
	}
	if (cudaMalloc((void **)&cudaRes, size)  == cudaErrorMemoryAllocation) {
		printf("Allocation memoire qui pose probleme (cudaRes) \n");
	}

	long dimBlock = blocksize;
	long dimGrid = tailleVecteur/blocksize;
	if ((tailleVecteur % blocksize) != 0) {
		dimGrid++;
	}

	




	// Ici on prend le temps de copy des vecteurs du CPU au GPU

startTimer;
	int res = cudaMemcpy(&cudaVec[0], &vecteur[0], size, cudaMemcpyHostToDevice);
stopTimer; 

	printf("Copy CPU -> GPU %d \n",res);
	int tpsCPUtoGPU = tpsCalcul;
	printf("Temps de copie CPU->GPU : %d \n", tpsCPUtoGPU);
	

	// Ici on prend le temps de calcul 
startTimer;
	add_vec_scalaire_gpu<<<dimGrid, dimBlock>>>(cudaVec, cudaRes, alpha, tailleVecteur);
stopTimer;


	int tpsCalculGPU = tpsCalcul;
	printf("Temps de calcul GPU : %d \n", tpsCalculGPU);
	

	// Ici on prend le temps de copy des vecteurs du GPU au CPU
startTimer;
	cudaMemcpy(&resultat[0], &cudaRes[0], size, cudaMemcpyDeviceToHost);
stopTimer;

	int tpsGPUtoCPU = tpsCalcul;
	printf("Temps de copie GPU->CPU : %d \n", tpsGPUtoCPU);
	

	




	/* Test bon fonctionnement */

	bool ok = true;
	int indice = -1;
	for (i= 0 ; i < tailleVecteur ; i++) {
/*		printf("Resultat GPU %d     Resultat CPU %d \n",resultat[i], vecteur[i]+alpha);	*/
		if (resultat[i] != vecteur[i] + alpha) {
			ok = false;
			if (indice ==-1) {
				indice = i;
			}
		}
	}
	printf("------ ");
	printf("dimGrid %ld dimBlock %ld ",dimGrid, dimBlock);
	if (ok) {
		printf("Resultat ok\n");
	} else {
		printf("resultat NON ok (%d)\n", indice);
	}

	int tpsTotal = tpsCPUtoGPU + tpsCalculGPU + tpsGPUtoCPU;
	printf("Vecteur %ld => Temps total pour le GPU = %d \n", tailleVecteur, tpsTotal);

	// Temps de calcul pour le CPU
startTimer;
	add_vec_scalaire_cpu (vecteur, resultat, alpha, tailleVecteur);
stopTimer;
	printf("Vecteur %ld => Temps total CPU : %ld \n", tailleVecteur, tpsCalcul);

	cudaFree(cudaVec);
	cudaFree(cudaRes);
}



