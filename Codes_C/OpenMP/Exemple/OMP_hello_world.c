#include <omp.h>
#include <stdio.h>

int main() {

	omp_set_num_threads(16);

	printf("Je suis le maitre, num thread % d, nbthreads %d !\n",
			omp_get_thread_num(), 
			omp_get_num_threads());

	#pragma omp parallel 
	{

		printf("Hello world from thread %d, nbthreads %d !\n",
			omp_get_thread_num(), 
			omp_get_num_threads());
	}


	printf("Je suis le maitre, num thread % d, nbthreads %d !\n",
			omp_get_thread_num(), 
			omp_get_num_threads());

	omp_set_num_threads(4);



	#pragma omp parallel 
	{

		printf("Hello world from thread %d, nbthreads %d !\n",
			omp_get_thread_num(), 
			omp_get_num_threads());
	}
	return 0;

}
