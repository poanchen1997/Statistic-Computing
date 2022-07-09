#include "random.h"


int main(){
    int n = 158;           //sample size
    int p = 51;            //number of variables
    int sample_time = 10000;
    char datafilename[] = "erdata.txt";
    char outputfile[] = "sample data.txt";
    char outputfile2[] = "covariance matrix for erdata.txt";
    char outputfile3[] = "covariance matrix for sample data.txt";

    //allocate the data matrix
    gsl_matrix* data = gsl_matrix_alloc(n,p);
    gsl_matrix* sample = gsl_matrix_alloc(sample_time, p); // allocate the sample data
    gsl_matrix* covar_origin = gsl_matrix_alloc(p, p);     // allocate the covariance matrix for erdata
    gsl_matrix* covar_sample = gsl_matrix_alloc(p, p);     // allocate the covariance matrix for sample data

    //read the data
    FILE* datafile = fopen(datafilename,"r");

    if(NULL==datafile){
        fprintf(stderr,"Cannot open data file [%s]\n",datafilename);
        return(0);
    }
    if(0!=gsl_matrix_fscanf(datafile,data)){ // record the data here
        fprintf(stderr,"File [%s] does not have the required format.\n",datafilename);
        return(0);
    }

    fclose(datafile);

    // program start    
    printf("Calculate the covariance matrix from original data.\n");
    makeCovariance(covar_origin, data, p, n);
    printmatrix(outputfile2, covar_origin);  // covariance of the original data

    printf("Sample from multivariate normal for %d times.\n", sample_time);
    gsl_rng* r;  // declare "r" in the below function
    randomMVN(sample_time, r, sample, covar_origin, p);
    printmatrix(outputfile, sample);            // the sample data

    printf("Calculate the covariance matrix from our sample data.\n");
    makeCovariance(covar_sample, sample, p, sample_time);
    printmatrix(outputfile3, covar_sample);  // covariance of the sample data

    //free memory
    gsl_matrix_free(data);
    gsl_matrix_free(sample);
    gsl_matrix_free(covar_origin);
    gsl_matrix_free(covar_sample);

    return(1);
}

