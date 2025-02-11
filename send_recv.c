#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#ifdef _OPENMP
	#include <omp.h>
#endif

void printM(double *M, int n, int g);
int checkSym(double *M, int n);
void matTranspose(double *M, double *T, int n);
int checkSymOMP(double *M, int t, int n);
void matTransposeOMP(double *M, double *T, int t, int n);
int checkSymMPI(double* M, int n, int rank, int size);
void matTransposeMPI(double* M, double* T, int n, int rank, int size);
void transpose(double* M, double* T, int n, int size);
int main(int argc, char** argv) {
   MPI_Init(&argc, &argv);
   int rank, size, begin, end, n, symmetry;
   double wt1, wt2;
   double *M;
   double *T;
   if(argc !=2 ){
      printf("Error: wrong number of parameters");
      MPI_Finalize();
      return 1;
   }
   n=atoi(argv[1]);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   
   if(size>n*n){
      if(rank==0) printf("error1: number of processors too big for the matrix\n");
      MPI_Finalize();
      return 1;
   }
   if(size%2 != 0 && size!=1){
      if(rank==0) printf("error2: uneven number of processors\n");
      MPI_Finalize();
      return 1;
   }
   M=(double*)malloc(n*n*sizeof(double));
   if(rank==0){
     T=(double*)malloc(n*n*sizeof(double));
  	 for(int i=0; i<n; i++){
      	for(int j=0; j<n; j++){
          	*(M + i*n + j)=1.0 + i*0.5 + j*0.1;
      	}
  	 }
   }
   if(rank==0) printf("\n%d x %d | %d ", n, n, size);
   
   if(rank==0){
  	wt1=MPI_Wtime();
    symmetry=checkSym(M, n);
  	matTranspose(M, T, n);
  	wt2=MPI_Wtime();
  	printf("| sequential: %f ", wt2-wt1);
   }
   if(rank==0){
  	wt1=MPI_Wtime();
   #ifdef _OPENMP
    symmetry=checkSymOMP(M, size, n);
  	matTransposeOMP(M, T, size, n);
   #endif
  	wt2=MPI_Wtime();
  	printf("| OMP: %f ", wt2-wt1);
    if(symmetry==0) 
       printf("| symmetric ");
    else 
       printf("| not symmetric ");
   }
   
   MPI_Barrier(MPI_COMM_WORLD);
   
   if(rank==0) wt1=MPI_Wtime();
   symmetry=checkSymMPI(M, n, rank, size);
   matTransposeMPI(M, T, n, rank, size);
   if(rank==0) wt2=MPI_Wtime();
   if(rank==0) {
      printf("| MPI: %f ", wt2-wt1);
      if(symmetry==0) 
         printf("| symmetric ");
      else 
         printf("| not symmetric ");
   }
   
   if(rank==0){
  	 int transposed=0;
  	 for(int i=0; i<n; i++){
      	for(int j=0; j<n; j++){
          	if( *(M + i*n + j) != *(T + j*n + i)) transposed=1;
      	}
  	 }
     if(transposed == 0)
  	   printf("| transposed");
     else 
       printf("not transposed");
   }
   if(rank==0){
  	free(M);
  	free(T);
   }
   MPI_Finalize();
   return 0;
}

void printM(double *M, int n, int g){
   for(int i=0; i<n; i++){
   	for(int j=0; j<g; j++){
       	printf("%2.2g\t", *(M + i*n +j));
   	}
   	printf("\n");
   }
}
int checkSym(double *M, int n){
   int sym=1;
   for(int i=0; i<n; i++){
    	for(int j=i+1; j<n; j++){
        	if(*(M + j*n + i) != *(M + i*n + j))
           	sym = 0;
    	}
   }
   return sym;
}
void matTranspose(double *M, double *T, int n){
   for (int i = 0; i < n; ++i){
    	for (int j = 0; j < n; ++j) {
         	*(T + j*n + i) = *(M + i*n + j);
    	}
   }
}
int checkSymOMP(double *M, int t, int n){
   int sym=0;
#pragma omp parallel num_threads(t) 
{
   #pragma omp for schedule(static, 8)
   for(int i=0; i<n; i++){
    	for(int j=i+1; j<n; j++){
        	if(*(M + j*n + i) != *(M + i*n + j))
           	sym = 1;
    	}
   }
}
   return sym;
}
void matTransposeOMP(double *M, double *T, int t, int n){
#pragma omp parallel num_threads(t) 
{
   #pragma omp for collapse(2) schedule(static, 8)
   for (int i = 0; i < n; ++i){
    	for (int j = 0; j < n; ++j) {
         	*(T + j*n + i) = *(M + i*n + j);
    	}
   }
}
}
int checkSymMPI(double* M, int n, int rank, int size){
  int symmetric=0, symmetry=0;
  MPI_Bcast(M, (n*n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if(n/size>=1){
     int begin=rank*n/size;
     int end=(rank+1)*n/size;
     if(n/size>1){ 
        begin=(rank*n/size)/2;
        end=begin + (n/size)/2;
     }
     for(int i=begin; i<end; i++){
   	for(int j=i+1; j<n; j++){
       	  if( *(M + j*n + i) != *(M + i*n + j)) symmetric=1;
   	}
     }
     if(end>1){
       for(int i=n-end; i<n-begin; i++){
   	  for(int j=i+1; j<n; j++){
       	    if( *(M + j*n + i) != *(M + i*n + j)) symmetric=1;
   	  }
       }
     }
   }else{
     for(int i=rank%n; i<rank%n + 1; i++){
     	  for(int j=0; j<n; j++){
            if(*(M + i + j*n) != *(M + j +i*n )) symmetric=1;
        }
     }
   }
   MPI_Reduce(&symmetric, &symmetry, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
   return symmetry;
}
void transpose(double* M, double* T, int n, int size){
   for(int i=0; i<n/size; i++){
  	  for(int j=0; j<n; j++){
     	  *(T + j*n/size + i)= *(M + i*n + j);
  	  }
   }
}
void matTransposeMPI(double* M, double* T, int n, int rank, int size){
   double* local_M=(double*)malloc(n*n*sizeof(double)/size);
   double* local_T=(double*)malloc(n*n*sizeof(double)/size);
   if(size<=n){
      if(rank==0){
  	    for(int i=1; i<size; i++){
     	    MPI_Send(M + (n*n/size)*i, n*n/size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
  	    }
      }else{
  	    MPI_Recv(local_M, n*n/size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  	    transpose(local_M, local_T, n, size);
  	    MPI_Send(local_T, n*n/size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      }
      if(rank==0){
  	    for(int i=0; i<n/size; i++){
     	        for(int j=0; j<n; j++)
        	   *(T + j*n + i) = *(M + i*n + j);
   	    }
  	    MPI_Datatype out_block_type;
  	    int sizes2[2]={n, n};
  	    int subsizes2[2]={n, n/size};
  	    int start2[2]={0, 0};
  	    MPI_Datatype temp_block_type;
  	    MPI_Type_create_subarray(2, sizes2, subsizes2, start2, MPI_ORDER_C, MPI_DOUBLE, &temp_block_type);
  	    MPI_Type_create_resized(temp_block_type, 0, 1*sizeof(double), &out_block_type);
  	    MPI_Type_commit(&out_block_type);
  	    MPI_Type_free(&temp_block_type);
  	    for(int i=1; i<size; i++){
     	      MPI_Recv(T + i*n/size, 1, out_block_type, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  	    }
  	    MPI_Type_free(&out_block_type);
     }
   }
   free(local_M);
   free(local_T);
}
