all :  VecteurCPU VecteurGPU ConfDevice

VecteurCPU : Vecteur.c
	gcc -o VecteurCPU Vecteur.c

VecteurGPU : Vecteur.cu
	nvcc -o VecteurGPU Vecteur.cu
ConfDevice : ConfDevice.cu
	nvcc -o ConfDevice ConfDevice.cu
clean :
	rm VecteurCPU VecteurGPU ConfDevice
