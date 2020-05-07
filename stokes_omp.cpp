#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include <iostream>
#include <fstream>
#include "utils.h"
#include <omp.h>
using namespace std;

//g++ -std=c++11 -O3 -fopenmp stokes_omp.cpp -o stokes_omp 

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&



//INITIALISING MATRICES *******************************************************************************************************************************************************
void initialise(int N, int N2, double *f,double *uxx, double h, double* p, double P_left, double P_right, double* D1, double* D2, double* D3)
{	
	
	
	// INITIALISE FORCE, INITIAL AND FINAL VELOCITY, RESIDUE AND LHS MATRICES 
	#pragma omp parallel for schedule(static) num_threads(8)
	for(long i=0;i<N2;i++){
		f[i]    = 0.0;
		uxx[i]	= 0.0;
	}

	
	//INITIALISING VELOCITY LAPLACIAN MATRICES ####################################
	#pragma omp parallel for schedule(static) num_threads(8)
	for(long j=0;j<(N2*N2);j++){	
		D1[j]    = 0.0;
		D2[j]    = 0.0;
		D3[j]	 = 0.0;
		
	}	

	// VELOCITY U LAPLACIAN U_xx + U_yy ##########################################

	#pragma omp parallel for schedule(static) num_threads(8)
	for(long j=0 ; j<N ; j=j+1){

		for(long i=0 ; i<N2 ; i=i+1){
			if(i==j){D1[i+j*N2] = 1;}
			
		}
	}

	#pragma omp parallel for schedule(static) num_threads(8)
	for(long j=N ; j<2*N ; j=j+1){

		for(long i=0 ; i<N2 ; i=i+1){
			if(i==j){
				D1[i+j*N2]=4;
				if((i+1)<N2){D1[i+1+j*N2] = -1;}
				if((i-1)>=0){D1[i-1+j*N2] = -1;}
				if((i+N)<N2){D1[(i+N) + j*N2] = -1;}
				if(j%N==0){
					if((i+1)<N2){D1[i+1+j*N2] = -2;}
					if((i-1)>=0){D1[i-1+j*N2] = 0;}
				}
				if((j+1)%N==0){
					if((i+1)<N2){D1[i+1+j*N2] = 0;}
					if((i-1)>=0){D1[i-1+j*N2] = -2;}
				}
			}
			
		}
	}	
		
	
	#pragma omp parallel for schedule(static) num_threads(8)
	for(long j=N2-N ; j<N2 ; j=j+1){

		for(long i=0 ; i<N2 ; i=i+1){
			if(i==j){D1[i+j*N2] = 1;}
			
		}
	}

	#pragma omp parallel for schedule(static) num_threads(8)
	for(long j=N2-(2*N) ; j<N2-N ; j=j+1){

		for(long i=0 ; i<N2 ; i=i+1){
			if(i==j){
				D1[i+j*N2]=4;
				if((i+1)<N2){D1[i+1+j*N2] = -1;}
				if((i-1)>=0){D1[i-1+j*N2] = -1;}
				if((i-N)>=0){D1[(i-N) + j*N2] = -1;}
				if(j%N==0){
					if((i+1)<N2){D1[i+1+j*N2] = -2;}
					if((i-1)>=0){D1[i-1+j*N2] = 0;}
				}
				if((j+1)%N==0){
					if((i+1)<N2){D1[i+1+j*N2] = 0;}
					if((i-1)>=0){D1[i-1+j*N2] = -2;}
				}
			}
			
		}
	}	

	#pragma omp parallel for schedule(static) num_threads(8)
	for(long j=(2*N) ; j<=N2-2*N ; j=j+1){

		for(long i=0 ; i<N2 ; i=i+1){
			if(i==j){
				D1[i+j*N2]=4;
				if((i+1)<N2){D1[i+1+j*N2] = -1;}
				if((i-1)>=0){D1[i-1+j*N2] = -1;}
				if((i-N)>=0){D1[(i-N) + j*N2] = -1;}
				if((i+N)<N2){D1[(i+N) + j*N2] = -1;}
				if(j%N==0){
					if((i+1)<N2){D1[i+1+j*N2] = -2;}
					if((i-1)>=0){D1[i-1+j*N2] = 0;}
				}
				if((j+1)%N==0){
					if((i+1)<N2){D1[i+1+j*N2] = 0;}
					if((i-1)>=0){D1[i-1+j*N2] = -2;}
				}
			}
			
		}
	}	

	

	// PRESSURE GRADIENT IN X DIRECTION P_x MATRIX D2 ################################################
	D2[0] = -0.5/h;
	D2[1] = 0.5/h;
	#pragma omp parallel for schedule(static) num_threads(8)
	for(long j=1 ; j<N2 ; j=j+1){

		for(long i=0 ; i<N2 ; i=i+1){

 	
			if(i==j){
				if((i+1)<N2 && (i-1)>=0){
					D2[(i+1)+j*N2] = 0.5/h;
					D2[(i-1)+j*N2] = -0.5/h;
				}

				if(((j+1)%N)==0){
					D2[i + j*N2] = 0.5/h;
					if((i-1)>=0){D2[(i-1) + j*N2] = -0.5/h;}
					if((i+1)<N2){D2[(i+1) + j*N2] = 0.0;}
				}
				if((j%N)==0){
					D2[i + j*N2] = -0.5/h;
					if((i+1)<N2){D2[(i+1) + j*N2] = +0.5/h;}
					if((i-1)>=0){D2[(i-1) + j*N2] = 0.0;}
				}

			}

		}

	}

	// PRESSURE LAPLACIAN P_xx + P_yy MATRIX D3  ##############################################################
	#pragma omp parallel for schedule(static) num_threads(8)
	for(long j=0 ; j<N2 ; j=j+1){

		for(long i=0 ; i<N2 ; i=i+1){
			if(i==j){

				if(j==0 || (j+1)%N==0){

					D3[i + j*N2]=1;
					if((j+1)<N2 && j>0){D3[(j+1) +(j+1)*N2]=1;}
				}
			}
		}
	}	

	#pragma omp parallel for schedule(static) num_threads(8)
	for(long j=1 ; j<N-1 ; j=j+1){
		for(long i=0 ; i<N2 ; i=i+1){
			if(i==j){
				D3[i+j*N2] = 4;
				if((i+1)<N2){D3[i+1 + j*N2]=-1;}
				if((i-1)>=0){D3[i-1 + j*N2]=-1;}
				if((i+N)<N2){D3[(i+N) + j*N2] = -2;}
				if(j==1){
					if((i-1)>=0){D3[i-1 + j*N2]=0;}
				}
				if(j==(N-2)){
					if((i+1)<N2){D3[i+1 + j*N2]=0;}
				}
				
			}

		}
	}
	#pragma omp parallel for schedule(static) num_threads(8)
	for(long j=N2-N+1 ; j<N2-1 ; j=j+1){
		for(long i=0 ; i<N2 ; i=i+1){
			if(i==j){
				D3[i+j*N2] = 4;
				if((i+1)<N2){D3[i+1 + j*N2]=-1;}
				if((i-1)>=0){D3[i-1 + j*N2]=-1;}
				if((i-N)>=0){D3[(i-N) + j*N2] = -2;}
				if(j==N2-N+1){
					if((i-1)>=0){D3[i-1 + j*N2]=0;}
				}
				if(j==(N2-2)){
					if((i+1)<N2){D3[i+1 + j*N2]=0;}
				}
				
			}

		}
	}

	for(long j=N ; j<N2-N ; j=j+1){
		if(j%N ==0 || (j+1)%N==0){continue;}
		for(long i=1 ; i<N2-1 ; i=i+1){
			if(i==j){
				D3[i+j*N2] = 4;
				if((i+1)<N2){D3[i+1+j*N2] = -1;}
				if((i-1)>=0){D3[i-1+j*N2] = -1;}
				if((i+N)<N2){D3[i+N+j*N2] = -1;}
				if((i-N)>=0){D3[i-N+j*N2] = -1;}
				if((j-1)%N==0){
					if((i-1)>=0){D3[i-1+j*N2] = 0;}
				}	
				if((j+2)%N==0){
					if((i+1)<N2){D3[i+1+j*N2] = 0;}
				}
			}
		}
	}
		


	// PRESSURE MATRIX ############################################################################
	#pragma omp parallel for schedule(static) num_threads(8)
	for(long j =0; j<N2; j++){
		p[0]=P_left; 
		p[1]=P_left; 
		if(j>0 && j%N==0){
			p[j]=P_left;
			p[j+1]=P_left;
			if(j-1>0){
				p[j-1]=P_right;
			}
		}
	}

}

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


//MATRIX MULTIPLICATION $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
void Mult(long m, double *a, double *b, double *c, double FL) {
  double sum = 0.0; 
  #pragma omp parallel for schedule(guided) num_threads(8)
  for (int j = 0; j < m; j++) {
	sum = 0.0;
	for (int i = 0; i < m; i++) {  		
			if(FL==0){sum = sum + a[i + m*j]*b[i];}
			if(FL==1){if(i!=j){sum = sum + a[i + m*j]*b[i];}}
     
      	}
	c[j]=sum;
	sum = 0.0;
  }
}


//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


//THE 2D JACOBI ALGORITHM $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
void jacobi(double *A, double *B, double*C, int N, int N2)
{	double *P1   		= (double *)malloc(N*N*sizeof(double)); // X_i matrix nth step
	double *P2   		= (double *)malloc(N*N*sizeof(double)); // X_i matrix (n+1)th step
	double *prod		= (double *)malloc(N*N*sizeof(double)); // After applying the differential matrix
	double *C1   		= (double *)malloc(N*N*sizeof(double)); // Local copy of RHS = known values coming from boundary conditions of LHS + force
	double *LHS   		= (double *)malloc(N*N*sizeof(double)); // Stores LHS
	double *R   		= (double *)malloc(N*N*sizeof(double)); // Matrix to store residues
	int REPS    		= 18000;				// Repetitions/Iterations
	double h 		= 1.00/(N+1);				// Discretisation step
	double res,res_init,flag 	= 0.0;				// Residue/Initial Residue and flag determines if multiplication is to be done with or without the A_ii element.

   	#pragma omp parallel for schedule(guided) num_threads(8)
	for(long j=0;j<N2;j++){
		prod[j] = 0.0;
		P1[j]   = 0.0;
		C1[j]	= C[j]+B[j];
		P2[j]   = 0.0;
		LHS[j]	= 0.0;
		R[j]	= 0.0;
	}



	for(int c=1;c<=REPS;c++){ 	// limit of # of iterations

		//#pragma omp parallel num_threads(8)
		//#pragma omp for collapse(2)

		flag = 1.0;		
		Mult(N2,A,P1,prod,flag);

		#pragma omp parallel for schedule(guided) num_threads(8)
		for(long j =0; j<N2;j++){		
			P2[j]= (1/A[j+j*N2])*(C1[j]-prod[j]);
			P1[j]= P2[j];
			prod[j]=0;
			P2[j]=0.0;
		}

		
		//COMPUTING RESIDUE ################################################
		
		flag = 0.0;
		Mult(N2,A,P1,prod,flag);

		#pragma omp parallel for schedule(guided) num_threads(8)
		for(long j =0; j<N2; j++){LHS[j]=prod[j]; prod[j]=0.0; }
		
		#pragma omp parallel for num_threads(18) reduction(+:res)		
		for (int i=0;i<(N*N);i++){
			R[i] = LHS[i] - C1[i];
	       		res = res + (R[i])*(R[i]);
		
		} 
		res = pow((res),0.5);
		//printf("Residue is %f  \n",res);

		// CHECKING FOR CONVERGENCE &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
		if(c==1){res_init = res;}

		if(c>=1){
		
			if(floor(log10(res_init/res))==6){break;printf("Min error\n");
			}
		}
		res = 0.0;	if(c%1000==0){printf("iter %d \n",c);}
	} 
	#pragma omp parallel for schedule(guided) num_threads(8)
	for(long j = 0;j<N2; j++){B[j]=P1[j];}

	// END OF JACOBI ITERATIONS $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

	

	// FREE ALL ALLOCATIONS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	free(P1);
	free(C1);
	free(P2);
	free(prod);
	free(LHS);
	free(R);
}
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

void evolve(int N, double h, double* U){
	long particles = 6*N;
	double *UX 	= (double *)malloc(N*sizeof(double));
	double *X 	= (double *)malloc((particles)*sizeof(double));
	double *Y 	= (double *)malloc((particles)*sizeof(double));
	double t_step	= 0.001;
	double step	= h;
	double time	= 1;
	int iter	= int(time/t_step);
	double u	=0.0;
	#pragma omp parallel for schedule(static) num_threads(8)
	for (long j=0;j<=N;j++){
		double sum = 0.0;
		for(long i=0;i<N;i++){
			sum = sum + U[i+j*N];
		}
		UX[j]=sum/N;
		sum =0.0;
	}
	
	
	ofstream fout3;
	fout3.open("evol.dat");
	fout3<<particles<<endl;
	fout3<<"\n";
	
	// INITIAL POSITIONS
	for(long i=0;i<6;i++){
		for(long j=0;j<N;j++){
			X[j+i*N]=(i/5.0)*(N-1)*step;
			Y[j+i*N]=j*step;
			fout3 << X[j+i*N] << " " << Y[j+i*N]<<endl;
		}
	}
	
	
	for(int k=0;k<=iter;k=k+1){
		fout3<<particles<<endl;
		fout3<<"\n";
		
		for(long i=0;i<6;i++){
			for(long j=0;j<N;j++){
		
				X[j+i*N] = (UX[j])*t_step + X[j+i*N];
				if(X[j+i*N]>(N-1)*step){X[j+i*N]=0+((N-1)*step-X[j+i*N]);}
				
				fout3 << X[j+i*N] << " " << Y[j+i*N]<<endl;
			}
			
		}
	
	}
	
	fout3.close();

}



//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
int main()

{  	
	

	//DECLARING ALL VARIABLES **********************************************************************************************************************************
	int N 			= 40; // Mesh of 40 * 40 grid points 
	int N2			= N*N;
	double *f  		= (double *)malloc(N*N*sizeof(double));   // body force matrix
	double *p  		= (double *)malloc(N*N*sizeof(double));   // pressure matrix
	double *px  		= (double *)malloc(N*N*sizeof(double));   // pressure gradient result P_x matrix
	double *uxx  		= (double *)malloc(N*N*sizeof(double));   // u matrix results after solving laplacian
	double P_left		= 200.0;				  // Pressure at pipe inlet
	double P_right		= 0.0; 					  // Pressure at pipe outlet
	double *D1 		= (double *)malloc(N2*N2*sizeof(double)); // D1 - velocity laplacian matrix
	double *D2 		= (double *)malloc(N2*N2*sizeof(double)); // D2 - pressure gradient matrix
	double *D3 		= (double *)malloc(N2*N2*sizeof(double)); // D3 - pressure laplacian matrix
	double h 		= 20.00/(N+1);				  // Mesh width
	double nu		= 2.00;					  // viscosity

	double t = omp_get_wtime();
	// INITIALISE ALL VARIABLES AND MATRICES $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	initialise(N, N2, f,uxx, h, p, P_left, P_right, D1, D2, D3); 
	printf("Initialisation done!\n");

	
	
	// COMPUTE PRESSURE DISTRIBBUTION $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	jacobi(D3,p,f,N,N2);
	printf("First jacobi done!\n");
	

	ofstream fout;		// WRITE THE RESULTS
	fout.open("pressure.dat");
	for(long j =0; j<N; j++){
		for(long i =0; i<N; i++){
			fout << i << " "<< j <<" "<< p[i+j*N] << "\n";
		}
	}
	fout.close();
	
	// COMPUTE PRESSURE GRADIENT $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	double flag = 0.0;
	Mult(N2,D2,p,px,flag);
	printf("First Mult done!\n");


	double *f1 = (double *)malloc(N*N*sizeof(double)); // Add P_x + f
	#pragma omp parallel for
	for(long j = 0;j<N2; j++){
		f1[j]=-(h*h/nu)*(f[j]+px[j]);
		if(j<N || j>=N2-N){f1[j]=0;}
	}
	
	// COMPUTE U VALUES $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	jacobi(D1,uxx,f1,N,N2);
	printf("Second jacobi done!\n");


	ofstream fout2;		// WRITE THE RESULTS
	fout2.open("Ux.dat");
	for(long i =0; i<N; i++){
		for(long j =0; j<N; j++){
			if(i%10==0){
			if(j>0 && j< N-1){
			fout2 << i << " "<< j << " " << uxx[i+j*N]/50 <<" "<< 0 << "\n";}
			if(j==0 || j== N-1){fout2 << i << " " << j << " " << 0 <<" "<< 0 << "\n";}
			}
			
		}
	}
	fout2.close();

	// MOL DYN OF PARTICLES $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$	
	//evolve(N,h,uxx);
	
	
	// FREE ALL ALLOCATIONS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	free(p);
	free(D1);
	free(D2);
	free(D3);
	free(px);
	free(uxx);
	free(f);
	free(f1);
	t = omp_get_wtime() - t;
	printf("Time take is %f \n",t);
	
	
}
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
