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
  	 MPI_Finalize();
  	 printf("error1: number of processors too big for the matrix\n");
  	 return 1;
   }
   if(size%2 != 0 && size!=1){
  	 MPI_Finalize();
  	 printf("error2: uneven number of processors\n");
  	 return 1;
   }
   M=(double*)malloc(n*n*sizeof(double));
   if(rank==0){
     T=(double*)malloc(n*n*sizeof(double));
  	 for(int i=0; i<n; i++){
      	for(int j=0; j<n; j++){
          	*(M + i*n + j)=1.0 * i + 0.5*j;
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
  	printf("| %d\n", transposed);
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
    	for(int j=0; j<n; j++){
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
   #pragma omp for collapse(2) schedule(static, 8)
   for(int i=0; i<n; i++){
    	for(int j=0; j<n; j++){
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
   if(n/size>1){
     int begin=rank*n/size;
     int end=(rank+1)*n/size;
     for(int i=begin; i<end; i++){
   	  for(int j=0; j<n; j++){
       	  if( *(M + j*n + i) != *(M + i*n + j)) symmetric=1;
   	  }
     }
   }else{
     int i, q=rank/n;
     i=rank%n;
     int block=n*n/size;
     int begin=(1 + q)*block;
     int end=begin + block;
     for(int j=begin; j<end; j++){
     	 if(*(M + j + i*n ) != *(M + i + (j)*n)) symmetric=1;
  	 }
   }
   MPI_Reduce(&symmetric, &symmetry, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
   return symmetry;
}
void matTransposeMPI(double* M, double* T, int n, int rank, int size){
   double wt1, wt2;
   double* local_M=(double*)malloc(n*n*sizeof(double)/size);
   int sizes[2]={n/size, n};
   int subsizes[2]={n/size, n};
   int start[2]={0, 0};
   int sizes2[2]={n, n/size};
   int subsizes2[2]={n, n/size};
   int start2[2]={0, 0};
   int disp[size];
   int num[size];
   MPI_Datatype block_type;
   MPI_Datatype send_block_type;
   MPI_Datatype out_block_type;
   MPI_Datatype local_block_type;
   if(n/size<1){
     	subsizes[0]=1;
     	subsizes[1]=(n*n)/size;
      sizes[0]=1;
      sizes[1]=n*n/size;
   }
   MPI_Type_create_subarray(2, sizes, subsizes, start, MPI_ORDER_C, MPI_DOUBLE, &block_type);
   MPI_Type_commit(&block_type);
   if(rank==0){
  	 int sizes[2]={n, n};
  	 int subsizes[2]={n/size, n};
  	 int start[2]={0, 0};
     if(n/size<1){
     	 subsizes[0]=1;
     	 subsizes[1]=(n*n)/size;
  	 }
  	 MPI_Datatype root_block_type;
  	 MPI_Type_create_subarray(2, sizes, subsizes, start, MPI_ORDER_C, MPI_DOUBLE, &root_block_type);
  	 MPI_Type_create_resized(root_block_type, 0, 1*sizeof(double), &send_block_type);
  	 MPI_Type_commit(&send_block_type);
  	 MPI_Type_free(&root_block_type);
   }
   if(rank==0){
    int block=n*n/size;
  	for(int i=0; i<size; i++){
      	disp[i]=i*block;
      	num[i]=1;
  	}
   }
   MPI_Scatterv(M, num, disp, send_block_type, local_M, 1, block_type, 0, MPI_COMM_WORLD);
   double* local_T=(double*)malloc(n*n*sizeof(double)/size);
   if(n/size<1){
     	subsizes2[0]=(n*n)/size;
     	subsizes2[1]=1;
      sizes2[0]=n*n/size;
      sizes2[1]=1;
  	}
   MPI_Type_create_subarray(2, sizes2, subsizes2, start2, MPI_ORDER_C, MPI_DOUBLE, &local_block_type);
   MPI_Type_commit(&local_block_type);
   if(rank==0){
  	 int sizes2[2]={n, n};
  	 int subsizes2[2]={n, n/size};
  	 int start2[2]={0, 0};
     if(n/size<1){
     	 subsizes2[0]=(n*n)/size;
     	 subsizes2[1]=1;
  	 }
  	 MPI_Datatype temp_block_type;
  	 MPI_Type_create_subarray(2, sizes2, subsizes2, start2, MPI_ORDER_C, MPI_DOUBLE, &temp_block_type);
  	 MPI_Type_create_resized(temp_block_type, 0, 1*sizeof(double), &out_block_type);
  	 MPI_Type_commit(&out_block_type);
  	 MPI_Type_free(&temp_block_type);
   }
   //if(rank==0) wt1=MPI_Wtime();
   if(n/size>=1){
  	for(int i=0; i<n/size; i++){
     	for(int j=0; j<n; j++){
        	*(local_T + j*n/size + i)= *(local_M + i*n + j);
     	}
  	}
   }else{
  	for(int j=0; j<n*n/size; j++){
     	*(local_T + j)= *(local_M + j);
  	}
   }
   /*if(rank==0){ 
      wt2=MPI_Wtime();
      printf("| transposition only rank %d: %f ", rank, wt2-wt1);
   }*/
   int disp_out[size];
   if(rank==0){
    int block=n/size;
  	for(int i=0; i<size; i++){
      	disp_out[i]=i*block;
      	num[i]=1;
  	}
  	if(n/size < 1){
     	int count=0, j=0;
     	for(int i=0; i<size; i++){
         	disp_out[i]=j + count*n * n*n/size;
         	count++;
         	if(count >= size/n){
            	j++;
            	count=0;
         	}
     	}
  	}
   }
   MPI_Gatherv(local_T, 1, local_block_type, T, num, disp_out, out_block_type, 0, MPI_COMM_WORLD);
   free(local_M);
   free(local_T);
   MPI_Type_free(&block_type);
   MPI_Type_free(&local_block_type);
   
   if(rank==0){
  	 MPI_Type_free(&send_block_type);
  	 MPI_Type_free(&out_block_type);
   }
}
