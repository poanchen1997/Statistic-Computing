#include "random.h"

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

// make covariance 
void makeCovariance(gsl_matrix* covX, gsl_matrix* X, int p, int n){
    int i, j, k;
    double data1[n], data2[n];  // I need "n" here to declare a length "n" list

    for (i = 0; i < p; i++){    // I need "p" here
        for (j = 0; j < p; j++){
            //generate the data1 and data2
            for (k = 0; k < n; k++){
                data1[k] = gsl_matrix_get(X, k, i);
                data2[k] = gsl_matrix_get(X, k, j);
            }

            // set the value to be covariance return by "gsl_stats_covariance"
            gsl_matrix_set(covX, i, j, 
                           gsl_stats_covariance(data1, 1, data2, 1, n));
        }
    }

    return;
}

// return the Cholesky matrix
gsl_matrix* makeCholesky(gsl_matrix* K, int p){
    int i, j;
    gsl_linalg_cholesky_decomp(K);

    // set the upper right triangular part of the metrix to be 0.
    for (i = 0; i < p-1; i++){ // I need "p" here.
        for (j = i+1; j < p; j++){
            gsl_matrix_set(K, i, j, 0);
        }
    }

    return(K);
}

// multivariate normal generator
void randomMVN(int sample_time, gsl_rng* mystream, gsl_matrix* samples, gsl_matrix* sigma, int p){
    // random set-up
    const gsl_rng_type* T;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    mystream = gsl_rng_alloc(T);

    // declare stuff
    int i, j;
    gsl_matrix* st_normal = gsl_matrix_alloc(p, 1);
    gsl_matrix* mult_normal = gsl_matrix_alloc(p, 1);
    gsl_vector* step_vector = gsl_vector_alloc(p);

    // set the cholesky matrix
    makeCholesky(sigma, p);

    // for loop for "sample_time" times
    for (i = 0; i < sample_time; i++){
        // sample from standard normal distribution
        for (j = 0; j < p; j++){  // I need "p" here to know how many numbers of sample from N(0, 1) should I generate.
            double u = gsl_ran_ugaussian(mystream);
            gsl_matrix_set(st_normal, j, 0, u);
        }

        // multiply the phi and st_normal
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, sigma, st_normal, 0.0, mult_normal);

        // receive the value in the mult_normal matrix
        gsl_matrix_get_col(step_vector, mult_normal, 0);

        // stack them to the output matrix
        gsl_matrix_set_row(samples, i, step_vector);
    }

    // free matrix and vector
    gsl_matrix_free(st_normal);
    gsl_matrix_free(mult_normal);
    gsl_vector_free(step_vector);

    // free rng
    gsl_rng_free(mystream);

    return;
}