#include "matrices.h"

//prints the elements of a matrix in a file
void printmatrix(char* filename,gsl_matrix* m)
{
	int i,j;
	double s;
	FILE* out = fopen(filename,"w");
	
	if(NULL==out) // test if the file open correctly
	{
		printf("Cannot open output file [%s]\n",filename);
		exit(1);
	}
	for(i=0;i<m->size1;i++)  // size1 is defined at the data structure of gsl_matrix m
	{
	    fprintf(out,"%.3lf",gsl_matrix_get(m,i,0));
		for(j=1;j<m->size2;j++)
		{
			fprintf(out,"\t%.3lf",
			gsl_matrix_get(m,i,j));
		}
		fprintf(out,"\n");
	}
	fclose(out);
	return;
}

//creates the transpose of the matrix m
gsl_matrix* transposematrix(gsl_matrix* m)
{
	int i,j;
	
	gsl_matrix* tm = gsl_matrix_alloc(m->size2,m->size1);
	
	for(i=0;i<tm->size1;i++)
	{
		for(j=0;j<tm->size2;j++)
		{
		  gsl_matrix_set(tm,i,j,gsl_matrix_get(m,j,i));
		}
	}	
	
	return(tm);
}

//calculates the product of a nxp matrix m1 with a pxl matrix m2
//returns a nxl matrix m
void matrixproduct(gsl_matrix* m1,gsl_matrix* m2,gsl_matrix* m)
{
	int i,j,k;
	double s;
	
	for(i=0;i<m->size1;i++) // size1 is the row?
	{
	  for(k=0;k<m->size2;k++) // size2 is the column?
	  {
	    s = 0;
	    for(j=0;j<m1->size2;j++)
	    {
	      s += gsl_matrix_get(m1,i,j)*gsl_matrix_get(m2,j,k);
	    }
	    gsl_matrix_set(m,i,k,s);
	  }
	}
	return;
}


//computes the inverse of a positive definite matrix
//the function returns a new matrix which contains the inverse
//the matrix that gets inverted is not modified
gsl_matrix* inverse(gsl_matrix* K)
{
	int j;
	
	gsl_matrix* copyK = gsl_matrix_alloc(K->size1,K->size1);
	if(GSL_SUCCESS!=gsl_matrix_memcpy(copyK,K))  // test if the copy correctly
	{
		printf("GSL failed to copy a matrix.\n");
		exit(1);
	}
	
	gsl_matrix* inverse = gsl_matrix_alloc(K->size1,K->size1);
	gsl_permutation* myperm = gsl_permutation_alloc(K->size1);
	
	if(GSL_SUCCESS!=gsl_linalg_LU_decomp(copyK,myperm,&j))
	{
		printf("GSL failed LU decomposition.\n");
		exit(1);
	}
	if(GSL_SUCCESS!=gsl_linalg_LU_invert(copyK,myperm,inverse))
	{
		printf("GSL failed matrix inversion.\n");
		exit(1);
	}
	gsl_permutation_free(myperm);
	gsl_matrix_free(copyK);
	
	return(inverse);
}

//creates a submatrix of matrix M
//the indices of the rows and columns to be selected are
//specified in the last four arguments of this function
gsl_matrix* MakeSubmatrix(gsl_matrix* M,
			  int* IndRow,int lenIndRow,
			  int* IndColumn,int lenIndColumn)
{
	int i,j;
	gsl_matrix* subM = gsl_matrix_alloc(lenIndRow,lenIndColumn);
	
	for(i=0;i<lenIndRow;i++)
	{
		for(j=0;j<lenIndColumn;j++)
		{
			gsl_matrix_set(subM,i,j, gsl_matrix_get(M,IndRow[i],IndColumn[j]));
		}
	}
	
	return(subM);
}


//computes the log of the determinant of a symmetric positive definite matrix
double logdet(gsl_matrix* K)
{
    int i;

	gsl_matrix* CopyOfK = gsl_matrix_alloc(K->size1,K->size2);
	gsl_matrix_memcpy(CopyOfK,K);
	gsl_permutation *myperm = gsl_permutation_alloc(K->size1);
	if(GSL_SUCCESS!=gsl_linalg_LU_decomp(CopyOfK,myperm,&i))
	{
		printf("GSL failed LU decomposition.\n");
		exit(1);
	}
	double logdet = gsl_linalg_LU_lndet(CopyOfK);
	gsl_permutation_free(myperm);
	gsl_matrix_free(CopyOfK);
	return(logdet);
}

// generate diagonal matrix
gsl_matrix* diag_alloc(int lenA){
	int i, j;

    gsl_matrix* mat = gsl_matrix_alloc(lenA, lenA);
    gsl_vector_view diag = gsl_matrix_diagonal(mat); // choose the element that is in diagonal
    gsl_matrix_set_all(mat, 0.0); 

	for(i=0;i<lenA;i++){
		for(j=0;j<lenA;j++){
			if (i == j){
				gsl_matrix_set(mat,i,j, 1);
			}
		}
	}

    return(mat);
}

// generate D data for vector A
gsl_matrix* MakeSubmatrix_whole(gsl_matrix* M,
			  int lenIndRow,
			  int* IndColumn,int lenIndColumn)
{
	int i,j;
	gsl_matrix* subM = gsl_matrix_alloc(lenIndRow,lenIndColumn);
	
	for(i=0;i<lenIndRow;i++)
	{
		for(j=0;j<lenIndColumn;j++)
		{
			gsl_matrix_set(subM,i,j, gsl_matrix_get(M, i, IndColumn[j] - 1));
		}
	}
	
	return(subM);
}

double marglik(gsl_matrix* data,int lenA,int* A, int n){ // I add a variable "n" here
	// compute the M_A part
	gsl_matrix* diag = diag_alloc(lenA);                 // create a diagonal matrix
    gsl_matrix* M_A = diag;                              // copy the diag, so that the original will not disappear
    gsl_matrix* D_A = MakeSubmatrix_whole(data, n, A, lenA); // I need "n" here
    gsl_matrix* D_AT = transposematrix(D_A);
    gsl_matrix* D_ATD_A = gsl_matrix_alloc(lenA, lenA);
    matrixproduct(D_AT, D_A, D_ATD_A);
    gsl_matrix_add(M_A, D_ATD_A);

	// log gamma part
	double upgamma = lgamma((n + lenA + 2)/2.0);
	double downgamma = lgamma((lenA + 2)/2.0);

	// long matrix
	int B[] = {1};
	int lenB = 1;
	gsl_matrix* D_1 = MakeSubmatrix_whole(data, n, B, lenB);  // I need "n" here
	gsl_matrix* D_1T = transposematrix(D_1);
	gsl_matrix* M_A_inv = inverse(M_A);
	gsl_matrix* longmatrix_result = gsl_matrix_alloc(lenB, lenB);
	gsl_matrix* product_step1 = gsl_matrix_alloc(lenB, lenA);
	gsl_matrix* product_step2 = gsl_matrix_alloc(lenB, lenA);
	gsl_matrix* product_step3 = gsl_matrix_alloc(lenB, n);

	matrixproduct(D_1T, D_A, product_step1);
	matrixproduct(product_step1, M_A_inv, product_step2);
	matrixproduct(product_step2, D_AT, product_step3);
	matrixproduct(product_step3, D_1, longmatrix_result);

	// get the result from two matrix
	double get_result1 = gsl_matrix_get(D_ATD_A, 0, 0);
	double get_result2 = gsl_matrix_get(longmatrix_result, 0, 0);

	// combine them all
	double result = upgamma - downgamma -
					0.5*logdet(M_A) - 
					((n + lenA + 2)/2.0)*log(1 + get_result1 - get_result2);

	// free memory
	gsl_matrix_free(diag);
	//gsl_matrix_free(M_A);  // I free it and the program aborted
	gsl_matrix_free(D_A);
	gsl_matrix_free(D_AT);
	gsl_matrix_free(D_ATD_A);
	gsl_matrix_free(D_1);
	gsl_matrix_free(D_1T);
	gsl_matrix_free(M_A_inv);
	gsl_matrix_free(longmatrix_result);
	gsl_matrix_free(product_step1);
	gsl_matrix_free(product_step2);
	gsl_matrix_free(product_step3);

	return(result);
}