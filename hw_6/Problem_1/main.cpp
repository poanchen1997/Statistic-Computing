#include "determinant.h"

int main(){
    char datafilename[] = "mybandedmatrix.txt";
    int n = 10;  // if we change the matrix in "mybandedmatrix.txt", we have to 
    int p = 10;  // change the row and column here. (n = row, p = column)

    // allocate the data matrix
    gsl_matrix* inputdata = gsl_matrix_alloc(n, p);

    // read the data
    FILE* datafile = fopen(datafilename,"r");

    if(NULL==datafile){
        fprintf(stderr,"Cannot open data file [%s]\n",datafilename);
        return(0);
    }
    if(0!=gsl_matrix_fscanf(datafile,inputdata)){ // record the data here
        fprintf(stderr,"File [%s] does not have the required format.\n",datafilename);
        return(0);
    }

    fclose(datafile);

    printf("The determinant of this %dX%d matrix is %.5lf.\n", n, p, getDeterminant(inputdata, n, p));

    // free memory
    gsl_matrix_free(inputdata);
    return(1);
}