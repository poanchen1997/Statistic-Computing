#include "matrices.h"

//allocates the memory for a matrix with 
//n rows and p columns
double ** allocmatrix(int n,int p)
{
	int i;
	double** m;
	
	m = new double*[n];
	for(i=0;i<n;i++)
	{
		m[i] = new double[p];
		memset(m[i],0,p*sizeof(double));
	}
	return(m);
}

//frees the memory for a matrix with n rows
void freematrix(int n,double** m)
{
	int i;
	
	for(i=0;i<n;i++)
	{
		delete[] m[i]; m[i] = NULL;
	}
	delete[] m; m = NULL;
	return;
}

//creates the copy of a matrix with n rows and p columns
void copymatrix(int n,int p,double** source,double** dest)
{
	int i,j;
	
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			dest[i][j] = source[i][j];
		}
	}
	return;
}

//reads from a file a matrix with n rows and p columns
void readmatrix(char* filename,int n,int p,double* m[])
{
	int i,j;
	double s;
	FILE* in = fopen(filename,"r");
	
	if(NULL==in)
	{
		printf("Cannot open input file [%s]\n",filename);
		exit(1);
	}
	for(i=0;i<n;i++)
	{
		for(j=0;j<p;j++)
		{
			fscanf(in,"%lf",&s);
			m[i][j] = s;
		}
	}
	fclose(in);
	return;
}

//prints the elements of a matrix in a file
void printmatrix(char* filename,int n,int p,double** m)
{
	int i,j;
	double s;
	FILE* out = fopen(filename,"w");
	
	if(NULL==out)
	{
		printf("Cannot open output file [%s]\n",filename);
		exit(1);
	}
	for(i=0;i<n;i++)
	{
		fprintf(out,"%.3lf",m[i][0]);
		for(j=1;j<p;j++)
		{
			fprintf(out,"\t%.3lf",m[i][j]);
		}
		fprintf(out,"\n");
	}
	fclose(out);
	return;
}

//creates the transpose of the matrix m
double** transposematrix(int n,int p,double** m)
{
	int i,j;
	
	double** tm = allocmatrix(p,n);
	
	for(i=0;i<p;i++)
	{
		for(j=0;j<n;j++)
		{
			tm[i][j] = m[j][i];
		}
	}	
	
	return(tm);
}

//calculates the dot (element by element) product of two matrices m1 and m2
//with n rows and p columns; the result is saved in m
void dotmatrixproduct(int n,int p,double** m1,double** m2,double** m)
{
	int i,j;
	
	for(i=0;i<n;i++)
	{
		for(j=0;j<p;j++)
		{
			m[i][j] = m1[i][j]*m2[i][j];
		}
	}
	
	return;
}

//calculates the product of a nxp matrix m1 with a pxl matrix m2
//returns a nxl matrix m
void matrixproduct(int n,int p,int l,double** m1,double** m2,double** m)
{
	int i,j,k;
	double s;
	
	for(i=0;i<n;i++)
	{
		for(k=0;k<l;k++)
		{
			s = 0;
			for(j=0;j<p;j++)
			{
				s += m1[i][j]*m2[j][k];
			}
			m[i][k] = s;
		}
	}
	return;
}

void set_mat_identity(int p, double *A)
{
 int i;

 for(i = 0; i < p * p; i++) A[i] = 0;
 for(i = 0; i < p; i++) A[i * p + i] = 1;
 return;
}

//computes the inverse of a symmetric positive definite matrix
void inverse(int p,double** m)
{
  int i,j,k;
  double* m_copy = (double*)malloc((p * p) * sizeof(double));
  double* m_inv = (double*)malloc((p * p) * sizeof(double));

  k=0;
  for(i=0;i<p;i++)
  {
     for(j=0;j<p;j++)
     {
        m_copy[k] = m[i][j];
        k++;
     }
  }

  set_mat_identity(p, m_inv);

  //-----  Use LAPACK  -------
  if(0!=(k=clapack_dposv(CblasRowMajor, CblasUpper, p, p, m_copy, p, m_inv, p)))
  {
    fprintf(stderr,"Something was wrong with clapack_dposv [%d]\n",k);
     exit(1);
  }
  //--------------------------

  k=0;
  for(i=0;i<p;i++)
  {
     for(j=0;j<p;j++)
     {
        m[i][j] = m_inv[k];
        k++;
     }
  }  

  free(m_copy);
  free(m_inv);

  return;
}


//computes the log of the determinant of a symmetric positive definite matrix
double logdet(int p,double** m)
{
	int i,j;
	char jobvl = 'N';
	char jobvr = 'N';
	int lda = p;
	double wr[2*p];
	double wi[2*p];
	double vl[p][p];
	int ldvl = p*p;
	double vr[p][p];
	int ldvr = p*p;
	double work[p*p];
	int lwork = p*p;
	double a[p][p];
	int info;
	
	for(i=0;i<p;i++)
	{
		for(j=0;j<p;j++)
		{
			a[i][j] = m[i][j];
		}
	}
	dgeev_(&jobvl,&jobvr,&p,(double*)a,&lda,(double*)wr,(double*)wi,(double*)vl, 
		  &ldvl,(double*)vr,&ldvr,(double*)work,&lwork,&info);

	if(0!=info)
	{
		printf("Smth wrong in the call of 'dgeev' error is [info = %d]\n",info);
		exit(1);
	}	   
	
	double logdet = 0;
	for(i=0;i<p;i++) logdet+=log(wr[i]);	
	return(logdet);
}

// Compute the marginal likelihood of regression
double marglik(int n,int p,double** data,int lenA,int* A){
	// n --> # of row
	// p --> # of variables
	// lenA --> # of indices
	// A --> indices of the variables present in the regression

	// M_A part
	double** m = allocmatrix(lenA, lenA);         // allocate a matrix for diagonal matrix
	double** dm = diagonalmatrix(lenA, lenA, m);  // create the diagonal matrix
	double** d = allocmatrix(n, lenA);            // allocate the D_A
	dmatrix(data, n, p, A, lenA, d);              // create D_A according to A
	double** D_T = transposematrix(n, lenA, d);   // create D_A^T
	double** D_TD = allocmatrix(lenA, lenA);      // allocate a matrix for D_A^T * D_A
	double** M_A = allocmatrix(lenA, lenA);       // allocate a matrix for M_A
	matrixproduct(lenA, n, lenA, D_T, d, D_TD);   // create D_TD
	M_A = addMatrix(lenA, lenA, dm, D_TD, M_A);   // create M_A
	
	// Log-gamma part and log-det part
	double uplgamma;                              // for the upper log-gamma in the formula
	double downlgamma;                            // for the down log-gamma in the formula
	double logdetM_A;                             // for the log-det part in the formula

	uplgamma = lgamma((n + lenA + 2)/2.0);        // if you just devide by 2 it will become integer
	downlgamma = lgamma((lenA + 2)/2.0);          // interger / float = float.
	logdetM_A = logdet(lenA, M_A);                

	// D_1^T * D^T part
	double** d_1 = allocmatrix(n, 1);             // allocate the D_1 matrix
	int B[] = {1};                                // the output variable is in index 1
	int lenB = 1;
	dmatrix(data, n, p, B, lenB, d_1);            // create D_1
	double** D_T_1 = transposematrix(n, lenB, d_1);// allocate D_1^T
	double** D_T_1D_1 = allocmatrix(lenB, lenB);  // allocate D_1^T * D_1
	matrixproduct(lenB, n, lenB, D_T_1, d_1, D_T_1D_1);  // create D_1^T * D_1

	// calculate that long matrix production
	double** longmatrix = allocmatrix(lenB, lenB); // allocate each step long matrix
	double** longmatrix_step1 = allocmatrix(lenB, lenA);
	double** longmatrix_step2 = allocmatrix(lenB, lenA);
	double** longmatrix_step3 = allocmatrix(lenB, n);
	double** M_A_inverse = M_A;                    // copy M_A_inverse from M_A
	inverse(lenA, M_A_inverse);                    // inverse the M_A
	matrixproduct(lenB, n, lenA, D_T_1, d, longmatrix_step1);  // create longmatrix
	matrixproduct(lenB, lenA, lenA, longmatrix_step1, M_A_inverse, longmatrix_step2);
	matrixproduct(lenB, lenA, n, longmatrix_step2, D_T, longmatrix_step3);
	matrixproduct(lenB, n, lenB, longmatrix_step3, d_1, longmatrix);

	double result;                                 // declare result
	result = uplgamma - downlgamma - (0.5*logdetM_A) - 
			((n + lenA + 2)/2.0)*log(1 + D_T_1D_1[0][0] - longmatrix[0][0]);

	// free matrix part
	freematrix(lenA , m); // first argument is row number
	freematrix(lenA , dm);
	freematrix(n , d);
	freematrix(lenA , D_T);
	freematrix(lenA , D_TD);
	freematrix(lenA , M_A);
	freematrix(n , d_1);
	freematrix(lenB , D_T_1);
	freematrix(lenB , D_T_1D_1);
	freematrix(lenB , longmatrix);
	freematrix(lenB , longmatrix_step1);
	freematrix(lenB , longmatrix_step2);
	freematrix(lenB , longmatrix_step3);
	// freematrix(lenA , M_A_inverse); // I free and the program aborted

	return(result);
}

//creates the diagonal matrix of the length lenA
double** diagonalmatrix(int n,int p,double** m) // m is the input matrix
{
	int i,j;
	
	double** tm = allocmatrix(p,n);
	
	for(i=0; i<p; i++){
		for(j=0; j<n; j++){
			if (i == j){
				tm[i][j] = 1;
			} 
		}
	}	
	
	return(tm);
}

// function to add two matrix
double** addMatrix(int n, int p, double** m1, double** m2, double** m_add){
	int i, j;

	for(i = 0; i < n; i++){
		for(j = 0; j < p; j++){
			m_add[i][j] = m1[i][j] + m2[i][j];
		}
	}

	return(m_add);
}

// get the value from the specific data
void dmatrix(double** input, int n, int p, int A[], int lenA, double** output){
	// input is the data matrix
	// output will be the data matrix only with the column in A
	int a, i, j;
	double s;
	
	for(i=0;i<n;i++){
		for(j=0; j<p; j++){
			for (a = 0; a < lenA; a++){
				if (j == A[a]-1){
					output[i][a] = input[i][j];
				}
			}
						
		}
	}

	return;
}