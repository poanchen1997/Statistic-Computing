#include "determinant.h"

double getDeterminant(gsl_matrix* inputMatrix, int n, int p){
    double answer, stepanswer;
    double sumstepanswer = 0;
    double leftup, leftdown, rightup, rightdown;
    double a_ij;
    int col, row, positive_negative, i;
    int n_minus = n - 1;
    int p_minus = p - 1;
    int n_ind[n_minus], p_ind[p], p_ind_2[p_minus];
    
    //gsl_matrix* submatrix()

    if (n == 1 && p == 1){
        answer = gsl_matrix_get(inputMatrix, 0, 0);
    } else if (n == 2 && p == 2){
        leftup = gsl_matrix_get(inputMatrix, 0, 0);
        leftdown = gsl_matrix_get(inputMatrix, 1, 0);
        rightup = gsl_matrix_get(inputMatrix, 0, 1);
        rightdown = gsl_matrix_get(inputMatrix, 1, 1);
        answer = leftup*rightdown - rightup*leftdown;
    } else {
        // make indice that "MakeSubmatrix" need;
        for (row = 0; row < n_minus; row++){
            n_ind[row] = row + 1;
        }
        for (col = 0; col < p; col++){
            p_ind[col] = col;
        }

        for (col = 0; col < p; col++){
            // two for loop to create the indices for "MakeSubMatrix"
            for(i=col; i<p_minus; i++){
                p_ind_2[i] = p_ind[i + 1];
            }
            for(i = 0; i < col; i++){
                p_ind_2[i] = p_ind[i];
            }

            a_ij = gsl_matrix_get(inputMatrix, 0, col);
            positive_negative = pow(-1, col+2);
            gsl_matrix* submatrix = MakeSubmatrix(inputMatrix, n_ind, n_minus, p_ind_2, p_minus);
            stepanswer = getDeterminant(submatrix, n_minus, p_minus);
            sumstepanswer += a_ij*positive_negative*stepanswer;

            // free matrix
            gsl_matrix_free(submatrix); 
        } 
        
        answer = sumstepanswer;
    }

    return(answer);
}

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

