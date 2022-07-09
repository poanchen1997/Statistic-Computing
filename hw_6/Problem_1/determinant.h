#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

// #include <gsl/gsl_math.h>
// #include <gsl/gsl_rng.h>
// #include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
// #include <gsl/gsl_statistics.h>
// #include <gsl/gsl_randist.h>
// #include <gsl/gsl_cdf.h>
// #include <gsl/gsl_blas.h>
// #include <gsl/gsl_permutation.h>
// #include <gsl/gsl_linalg.h>
// #include <gsl/gsl_sort_double.h>
// #include <gsl/gsl_sort_vector.h>
// #include <gsl/gsl_errno.h>

double getDeterminant(gsl_matrix* inputMatrix, int n, int p);
gsl_matrix* MakeSubmatrix(gsl_matrix* M,
			  int* IndRow,int lenIndRow,
			  int* IndColumn,int lenIndColumn);