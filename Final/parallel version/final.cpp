#include "final.h"

//prints the elements of a matrix in a file
void printmatrix(char* filename,gsl_matrix* m){
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

// generate D data for vector A (abbreviate the row arguments)
void MakeSubmatrix_whole(gsl_matrix* M,
			  int lenIndRow,
			  int* IndColumn,int lenIndColumn, gsl_matrix* subM){
	int i,j;
	
	for(i=0;i<lenIndRow;i++){
		for(j=0;j<lenIndColumn;j++){
			gsl_matrix_set(subM,i,j, gsl_matrix_get(M, i, IndColumn[j] - 1));
		}
	}
	
	return;
}

// the inverse of the logit function
double inverseLogit(double x){
    return(exp(x)/(1+exp(x)));
}

// function for the computation of the Hessian
double inverseLogit2(double x){
    return(exp(x)/pow(1+exp(x), 2));
}

// computes pi_i = P(y_i = 1| x_i)
void getPi(gsl_matrix* x, gsl_matrix* beta, gsl_matrix* pi){
    int i;
    // setting x0
    gsl_matrix* x0 = gsl_matrix_alloc(x->size1, 2);

    // setting first column to be 1 and second column to be x.
    for(i = 0; i < x->size1;i++){
        gsl_matrix_set(x0, i, 0, 1);
        gsl_matrix_set(x0, i, 1, gsl_matrix_get(x, i, 0));
    }

    // multiply two matrix 148*2 * 2*1 = 148*1
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, x0, beta, 0.0, pi);
    
    // insert the inverselogit value to the pi matrix
    for (i = 0; i < pi->size1; i++){
        gsl_matrix_set(pi, i, 0, inverseLogit(gsl_matrix_get(pi, i, 0)));
    }

    // free matrix
    gsl_matrix_free(x0);

    return;
}

// another function for the computation of the Hessian
void getPi2(gsl_matrix* x, gsl_matrix* beta, gsl_matrix* pi2){
    int i;
    // setting x0
    gsl_matrix* x0 = gsl_matrix_alloc(x->size1, 2);

    // setting first column to be 1 and second column to be x.
    for(i = 0; i < x->size1;i++){
        gsl_matrix_set(x0, i, 0, 1);
        gsl_matrix_set(x0, i, 1, gsl_matrix_get(x, i, 0));
    }

    // multiply two matrix 148*2 * 2*1 = 148*1
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, x0, beta, 0.0, pi2);

    // insert the inverselogit2 value to the pi2 matrix
    for (i = 0; i < pi2->size1; i++){
        gsl_matrix_set(pi2, i, 0, inverseLogit2(gsl_matrix_get(pi2, i, 0)));
    }

    gsl_matrix_free(x0);
    return;
}

// compute logistic log-likelihood 
double logisticLoglik(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta){
    int i;

    // compute pi
    gsl_matrix* pi = gsl_matrix_alloc(x->size1, 1);
    getPi(x, beta, pi);

    double answer = 0.0;

    // create y*log(pi)
    gsl_matrix* logpi = gsl_matrix_alloc(pi->size1, 1);
    gsl_matrix* ylogpi = gsl_matrix_alloc(pi->size1, 1);
    for (i = 0; i < logpi->size1; i++){
        gsl_matrix_set(logpi, i, 0, log(gsl_matrix_get(pi, i, 0)));
    }
    for (i = 0; i < ylogpi->size1; i++){
        gsl_matrix_set(ylogpi, i, 0, gsl_matrix_get(logpi, i, 0)*gsl_matrix_get(y, i, 0));
    }

    // create (1-y)*log(1-pi)
    gsl_matrix* mlogpi = gsl_matrix_alloc(pi->size1, 1);
    gsl_matrix* mymlogpi = gsl_matrix_alloc(pi->size1, 1);
    for (i = 0; i < mlogpi->size1; i++){
        gsl_matrix_set(mlogpi, i, 0, log(1-gsl_matrix_get(pi, i, 0)));
    }
    for (i = 0; i < mymlogpi->size1; i++){
        gsl_matrix_set(mymlogpi, i, 0, gsl_matrix_get(mlogpi, i, 0)*(1-gsl_matrix_get(y, i, 0)));
    }

    // sum them all
    for (i = 0; i < ylogpi->size1; i++){
        answer += gsl_matrix_get(ylogpi, i, 0);
        answer += gsl_matrix_get(mymlogpi, i, 0);
    }

    // free matrix
    gsl_matrix_free(pi);
    gsl_matrix_free(logpi);
    gsl_matrix_free(ylogpi);
    gsl_matrix_free(mlogpi);
    gsl_matrix_free(mymlogpi);
    return(answer);
}

// calculates l^*(\beta_0, \beta_1)
double lStar(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta){
    double answer = 0;

    // compute beta0^2 and beta1^2
    double beta02 = pow(gsl_matrix_get(beta, 0, 0), 2);
    double beta12 = pow(gsl_matrix_get(beta, 1, 0), 2);

    answer = -(beta02 + beta12)/2 + logisticLoglik(y, x, beta);

    return (answer);
}

// obtain the gradient for Newton-Raphson
void getGradient(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, gsl_matrix* gradient){
    int i;
    // set gradient to zero
    gsl_matrix_set_zero(gradient);

    // set pi
    gsl_matrix* pi = gsl_matrix_alloc(x->size1, 1);
    getPi(x, beta, pi);

    // get beta0 and beta1
    double beta0 = gsl_matrix_get(beta, 0, 0);
    double beta1 = gsl_matrix_get(beta, 1, 0);

    // set ans1 & ans2
    double ans1 = -beta0;
    for(i = 0; i < pi->size1; i++){
        ans1 += (gsl_matrix_get(y, i, 0) - gsl_matrix_get(pi, i, 0));
    }

    double ans2 = -beta1;
    for(i = 0; i < pi->size1; i++){
        ans2 += (gsl_matrix_get(y, i, 0) - gsl_matrix_get(pi, i, 0))*gsl_matrix_get(x, i, 0);
    }

    // insert the ans to gradient
    gsl_matrix_set(gradient, 0, 0, ans1);
    gsl_matrix_set(gradient, 1, 0, ans2);

    // free matrix
    gsl_matrix_free(pi);
    return;
}

// obtain the Hessian for Newton-Raphson
void getHessian(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, gsl_matrix* Hessian){
    int i;

    // get pi2
    gsl_matrix* pi2 = gsl_matrix_alloc(x->size1, 1);
    getPi2(x, beta, pi2);   

    // compute value for hessian
    double ans1 = 1;
    double ans23 = 0;
    double ans4 = 1;
    for (i = 0; i < pi2->size1; i++){
        ans1 += gsl_matrix_get(pi2, i, 0);
        ans23 += (gsl_matrix_get(pi2, i, 0)*gsl_matrix_get(x, i, 0));
        ans4 += (gsl_matrix_get(pi2, i, 0)*pow(gsl_matrix_get(x, i, 0), 2));
    }

    // insert the value to Hessian matrix
    gsl_matrix_set(Hessian, 0, 0, -ans1);
    gsl_matrix_set(Hessian, 0, 1, -ans23);
    gsl_matrix_set(Hessian, 1, 0, -ans23);
    gsl_matrix_set(Hessian, 1, 1, -ans4);

    // free matrix
    gsl_matrix_free(pi2);

    return;
}

// the function to compute the inverse matrix
void inverse(gsl_matrix* K, gsl_matrix* inverse)
{
	int j;
	
	gsl_matrix* copyK = gsl_matrix_alloc(K->size1,K->size1);
    // test if the copy correctly
	if(GSL_SUCCESS!=gsl_matrix_memcpy(copyK,K))  
	{
		printf("GSL failed to copy a matrix.\n");
		exit(1);
	}
	
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

    // free matrix
	gsl_permutation_free(myperm);
	gsl_matrix_free(copyK);
	
	return;
}

// this function implements our own Newton-Raphson procedure, result is 2X1 matrix for beta
void getcoefNR(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta){
    double epsilon = 0.0000000001;
    int i;
    
    // initial the beta to zero
    gsl_matrix_set_zero(beta);

    // current lStar
    double currentLStar = lStar(y, x, beta);
    int iteration = 0;

    // declare the matrix we use in the while function
    //gsl_matrix* stepanswer = gsl_matrix_alloc(2, 1);
    gsl_matrix* hessian = gsl_matrix_alloc(2, 2);
    gsl_matrix* invhessian = gsl_matrix_alloc(2, 2);
    gsl_matrix* gradient = gsl_matrix_alloc(2, 1);
    gsl_matrix* invhesgre = gsl_matrix_alloc(2, 1);
    gsl_matrix* new_beta = gsl_matrix_alloc(2, 1);
    gsl_matrix* diffmatrix = gsl_matrix_alloc(2, 1);

    while(1){
        iteration += 1;
        // compute the hessian matrix
        getHessian(y, x, beta, hessian);
        inverse(hessian, invhessian);
        // compute the gradient matrix
        getGradient(y, x, beta, gradient);
        // multiply the above two matrix
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, invhessian, gradient, 0.0, invhesgre);
        
        // new beta iteration
        gsl_matrix_set(new_beta, 0, 0, 
                        (gsl_matrix_get(beta, 0, 0) - gsl_matrix_get(invhesgre, 0, 0)));
        gsl_matrix_set(new_beta, 1, 0, 
                        (gsl_matrix_get(beta, 1, 0) - gsl_matrix_get(invhesgre, 1, 0)));
        // new lstar iteration
        double newLStar = lStar(y, x, new_beta);
        
        // error condition --> it seems something will error and result in stopping at iteration 5.
        // might because of the format of the newLStar and currentLStar
        // is seems when going to like more than 20 decimal, newLStar are bigger than currentLStar
        // so I round them to a smaller number of decimal.
        if (round(newLStar*1000000)/1000000 < round(currentLStar*1000000)/1000000){
            printf("CODING ERROR!! :: %d :: %.20lf :: %.20lf.\n", iteration, newLStar, currentLStar);
            break;
        }
        
        // stop if the log-likelihood does not improve by too much
        gsl_matrix_set(diffmatrix, 0, 0, 
                        fabs(gsl_matrix_get(new_beta, 0, 0) - gsl_matrix_get(beta, 0, 0)));
        gsl_matrix_set(diffmatrix, 1, 0, 
                        fabs(gsl_matrix_get(new_beta, 1, 0) - gsl_matrix_get(beta, 1, 0)));
        if (gsl_matrix_max(diffmatrix) < epsilon){
            break;
        }

        // update the beta and currentLStar
        for(i = 0; i < beta->size1; i++){
            gsl_matrix_set(beta, i, 0, gsl_matrix_get(new_beta, i, 0));
        }
        currentLStar = newLStar;
    }

    // free matrix
    gsl_matrix_free(hessian);
    gsl_matrix_free(invhessian);
    gsl_matrix_free(gradient);
    gsl_matrix_free(invhesgre);
    gsl_matrix_free(new_beta);
    gsl_matrix_free(diffmatrix);

    return;
}

// performs one iteration for the Metropolis-Hastings algorithm
void mhLogisticRegression(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, 
                          gsl_matrix* beta, gsl_matrix* invNegHessian, gsl_matrix* beta_output){
    int i;

    // use beta and invNegHessian to generate one example for multivariate normal
    // store in betaCandidate(1X2)
    gsl_matrix* betaCandidate = gsl_matrix_alloc(1, 2);
    gsl_matrix* transbetaCandidate = gsl_matrix_alloc(2, 1);

    mvrnorm(1, mystream, beta, invNegHessian, betaCandidate);
    transposematrix(betaCandidate, transbetaCandidate);  // transpose it to 2X1
    
    // lstar value
    double currentLStar = lStar(y, x, beta);
    double candidateLStar = lStar(y, x, transbetaCandidate);

    // pop out condition 1:
    if (candidateLStar >= currentLStar){
        for (i = 0; i < betaCandidate->size1; i++){
            gsl_matrix_set(beta_output, i, 0, gsl_matrix_get(transbetaCandidate, i, 0));
        }

        gsl_matrix_free(betaCandidate);
        gsl_matrix_free(transbetaCandidate);
        return;
    }
    
    // pop out condition 2:
    double u = gsl_rng_uniform(mystream);
    if (u <= exp(candidateLStar - currentLStar)){
        for (i = 0; i < betaCandidate->size1; i++){
            gsl_matrix_set(beta_output, i, 0, gsl_matrix_get(transbetaCandidate, i, 0));
        }
        
        gsl_matrix_free(betaCandidate);
        gsl_matrix_free(transbetaCandidate);
        return;
    }

    // reject the move and stay at the current state
    for (i = 0; i < betaCandidate->size1; i++){
        gsl_matrix_set(beta_output, i, 0, gsl_matrix_get(beta, i, 0));
    }

    gsl_matrix_free(betaCandidate);
    gsl_matrix_free(transbetaCandidate);
    return;
}

// generate "sampletimes" times sample from multivariate normal(beta, sigma) store in sample(sampletimesX2)
void mvrnorm(int sampletimes, gsl_rng* mystream, gsl_matrix* beta, gsl_matrix* sigma, gsl_matrix* sample){
    int i, j;
    gsl_matrix* st_normal = gsl_matrix_alloc(sigma->size1, 1);
    gsl_matrix* mult_normal = gsl_matrix_alloc(sigma->size1, 1);
    gsl_vector* step_vector = gsl_vector_alloc(sigma->size1);
    gsl_matrix* phi = gsl_matrix_alloc(sigma->size1, sigma->size2);

    makeCholesky(sigma, phi);
    for (j = 0; j < sampletimes; j++){
        for(i = 0; i < phi->size1; i++){
            double u = gsl_ran_ugaussian(mystream);
            gsl_matrix_set(st_normal, i, 0, u);
        }
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, phi, st_normal, 0.0, mult_normal);
        for (i = 0; i < phi->size1; i++){
            gsl_matrix_set(mult_normal, i, 0, 
                            gsl_matrix_get(mult_normal, i, 0) + gsl_matrix_get(beta, i, 0));
        }
        // receive the value in the mult_normal matrix
        gsl_matrix_get_col(step_vector, mult_normal, 0);
        
        // stack them to the output matrix
        gsl_matrix_set_row(sample, j, step_vector);
    }

    // free matrix and vector
    gsl_matrix_free(st_normal);
    gsl_matrix_free(mult_normal);
    gsl_vector_free(step_vector);
    gsl_matrix_free(phi);
    
    return;
}

// return the Cholesky matrix
void makeCholesky(gsl_matrix* K, gsl_matrix* out){
    int i, j;
    for (i=0 ; i<out->size1; i++){
        for(j = 0; j<out->size2; j++){
            gsl_matrix_set(out, i, j, gsl_matrix_get(K, i, j));
        }
    }
    gsl_linalg_cholesky_decomp(out);

    // set the upper right triangular part of the metrix to be 0.
    for (i = 0; i < out->size1-1; i++){ 
        for (j = i+1; j < out->size2; j++){
            gsl_matrix_set(out, i, j, 0);
        }
    }
    
    return;
}

// transpose the matrix
void transposematrix(gsl_matrix* m, gsl_matrix* tm)
{
	int i,j;
	
	for(i=0;i<tm->size1;i++){
		for(j=0;j<tm->size2;j++){
		  gsl_matrix_set(tm,i,j,gsl_matrix_get(m,j,i));
		}
	}	
	
	return;
}

// get Laplace approximation 
double getLaplaceApprox(gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode){
    double maxLogLik = logisticLoglik(y, x, betaMode);

    // get beta_0^2 and beta_1^2
    double beta02 = pow(gsl_matrix_get(betaMode, 0, 0), 2);
    double beta12 = pow(gsl_matrix_get(betaMode, 1, 0), 2);

    gsl_matrix* hessians = gsl_matrix_alloc(2, 2);
    getHessian(y, x, betaMode, hessians);

    // compute value
    double logmarglik = -(beta02+beta12)/2 + maxLogLik - 0.5*getLogDeterminant(hessians, 2, 2);

    // free matrix
    gsl_matrix_free(hessians);

    return(logmarglik);
}

// get log determinent for matrix
double getLogDeterminant(gsl_matrix* inputMatrix, int n, int p){
    double answer, stepanswer, sumstepanswer = 0;
    double leftup, leftdown, rightup, rightdown;
    double a_ij;
    int col, row, positive_negative, i, n_minus = n - 1, p_minus = p - 1;
    int n_ind[n_minus], p_ind[p], p_ind_2[p_minus];
    

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

        gsl_matrix* submatrix = gsl_matrix_alloc(n_minus, p_minus);
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
            MakeSubmatrix(inputMatrix, n_ind, n_minus, p_ind_2, p_minus, submatrix);
            stepanswer = getLogDeterminant(submatrix, n_minus, p_minus);
            sumstepanswer += a_ij*positive_negative*stepanswer;
        } 
        // free matrix
        gsl_matrix_free(submatrix); 
        
        answer = sumstepanswer;
    }

    return(log(answer));
}

void MakeSubmatrix(gsl_matrix* M,
			  int* IndRow,int lenIndRow,
			  int* IndColumn,int lenIndColumn, gsl_matrix* subM){
	int i,j;
	
	for(i=0;i<lenIndRow;i++){
		for(j=0;j<lenIndColumn;j++){
			gsl_matrix_set(subM,i,j, gsl_matrix_get(M,IndRow[i],IndColumn[j]));
		}
	}
	
	return;
}

// get the posterior means
void getPosteriorMeans(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, 
                        gsl_matrix* betaMode, int NumberOfIteration, gsl_matrix* betaBayes){
    int i, j, k;

    gsl_matrix* betaCurrent = gsl_matrix_alloc(2, 1);
    for (i = 0; i < betaCurrent->size1; i++){
        gsl_matrix_set(betaCurrent, i, 0, gsl_matrix_get(betaMode, i, 0));
    }

    // compute negative inverse hessian
    gsl_matrix* hessian = gsl_matrix_alloc(2, 2);
    getHessian(y, x, betaMode, hessian);
    gsl_matrix* invNegHessian = gsl_matrix_alloc(2, 2);
    inverse(hessian, invNegHessian);
    gsl_matrix_set(invNegHessian, 0, 0, -gsl_matrix_get(invNegHessian, 0, 0));
    gsl_matrix_set(invNegHessian, 1, 0, -gsl_matrix_get(invNegHessian, 1, 0));
    gsl_matrix_set(invNegHessian, 0, 1, -gsl_matrix_get(invNegHessian, 0, 1));
    gsl_matrix_set(invNegHessian, 1, 1, -gsl_matrix_get(invNegHessian, 1, 1));

    gsl_matrix* invNegHessians = gsl_matrix_alloc(2, 2);

    double ans0 = 0.0, ans1 = 0.0;
    for (i = 0; i < NumberOfIteration; i++){
        for (j = 0; j < invNegHessian->size1; j++){
            for (k = 0; k < invNegHessian->size1; k++){
                gsl_matrix_set(invNegHessians, j, k, 
                                gsl_matrix_get(invNegHessian, j, k));
            }
        }
        mhLogisticRegression(mystream, y, x, betaCurrent, invNegHessians, betaCurrent);
        ans0 += gsl_matrix_get(betaCurrent, 0, 0);
        ans1 += gsl_matrix_get(betaCurrent, 1, 0);
    }
    ans0 /= NumberOfIteration;
    ans1 /= NumberOfIteration;
    gsl_matrix_set(betaBayes, 0, 0, ans0);
    gsl_matrix_set(betaBayes, 1, 0, ans1);

    // free matrix
    gsl_matrix_free(betaCurrent);
    gsl_matrix_free(hessian);
    gsl_matrix_free(invNegHessian);
    gsl_matrix_free(invNegHessians);

    return;
}

// get Monte Carlo intergration
double getMonteCarloIntegration(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x){
    // setting beta
    gsl_matrix* beta = gsl_matrix_alloc(2, 1);  // allocate a 2X1 matrix
    gsl_matrix_set_zero(beta);                  // initial the beta to 0

    // setting Sigma
    gsl_matrix* Sigma = gsl_matrix_alloc(2, 2); // allocate a 2X2 matrix
    gsl_matrix_set_zero(Sigma);
    gsl_matrix_set(Sigma, 0, 0, 1);
    gsl_matrix_set(Sigma, 1, 1, 1);

    // generate 10000 sample for N(beta, Sigma)
    gsl_matrix* sample = gsl_matrix_alloc(10000, 2);
    mvrnorm(10000, mystream, beta, Sigma, sample);   // store the result in the sample
    
    // calculate the lstar and exponential it
    gsl_matrix* betaj = gsl_matrix_alloc(2, 1);
    gsl_vector* lvalue = gsl_vector_alloc(10000);
    double ans, stepans = 0.0;
    int i;
    for (i = 0; i < sample->size1; i++){
        // set beta_j
        gsl_matrix_set(betaj, 0, 0, gsl_matrix_get(sample, i, 0));
        gsl_matrix_set(betaj, 1, 0, gsl_matrix_get(sample, i, 1));
        
        // store the step answer in the lvalue
        gsl_vector_set(lvalue, i, logisticLoglik(y, x, betaj));
    }

    // find max value in the vector lvalue
    double maxloglikVec = gsl_vector_max(lvalue);
    for(i = 0; i < lvalue->size; i++){
        gsl_vector_set(lvalue, i, exp(gsl_vector_get(lvalue, i)-maxloglikVec));
    }
    for(i = 0; i < lvalue->size; i++){
        stepans += gsl_vector_get(lvalue, i);
    }
    stepans/= 10000;
    ans = log(stepans) + maxloglikVec;

    // free matrix and vector
    gsl_matrix_free(beta);
    gsl_matrix_free(Sigma);
    gsl_matrix_free(sample);
    gsl_matrix_free(betaj);
    gsl_vector_free(lvalue);

    return(ans);
}

// add a new set of value to the linklist
void AddRegression(int nMax_in_list, LPLinklist linklist, int indexA, 
                    double LlogmarglikA, double MlogmarglikA, gsl_matrix* beta){
    int i; 
    LPLinklist p = linklist;
    LPLinklist pnext = p->Next;

    while(NULL != pnext){
        // go to the next element in the list if the current
        // average(LlogmarglikA and MlogmarglikA) are bigger than the next one
        if ((pnext->LlogmarglikA+pnext->MlogmarglikA)/2 > (LlogmarglikA+MlogmarglikA)/2){
            p = pnext;
            pnext = p->Next;
        }
        else{
            break;
        }
    }

    // create a new element of the list
    LPLinklist newp = new Linklist;
    newp->indexA = indexA;
    newp->LlogmarglikA = LlogmarglikA;
    newp->MlogmarglikA = MlogmarglikA;
    newp->beta = gsl_matrix_alloc(2, 1);

    for (i = 0; i<2;i++){
        gsl_matrix_set(newp->beta, i, 0, gsl_matrix_get(beta, i, 0));
    }

    // insert the new element in the list
    p->Next = newp;
    newp->Next = pnext;

    // print the insert message
    printf("inserted variable %d to the linklist.\n", indexA);

    // delete part if linklist has already has "nMax_in_list" elements
    int numofindex = getCount(linklist);
    if (numofindex > nMax_in_list){
        DeleteLastRegression(linklist);
    }

    return;
}

// function to get the length of the link list
int getCount(LPLinklist linklist){
    int count = 0;                 // Initialize count
    LPLinklist p = linklist;     // Initialize link list
    while (p->Next != NULL){
        count++;
        p = p->Next;
    }
    return count;
}

//this function deletes the last element of the list
//with the head "regressions"
//again, the head is not touched
void DeleteLastRegression(LPLinklist linklist){
  //this is the element before the first regression
  LPLinklist pprev = linklist;
  //this is the first regression
  LPLinklist p = linklist->Next;

  //if the list does not have any elements, return
  if(NULL == p){
    return;
  }

  //the last element of the list is the only
  //element that has the "Next" field equal to NULL
  while(NULL != p->Next){
    pprev = p;
    p = p->Next;
  }
  
  //now "p" should give the last element
  //delete it
  //delete[] p->A;
  gsl_matrix_free(p->beta);
  p->Next = NULL;
  delete p;

  //now the previous element in the list
  //becomes the last element
  pprev->Next = NULL;

  return;
}

//this function saves the regressions in the list with
//head "regressions" in a file with name "filename"
void SaveRegressions(char* filename,LPLinklist linklist){
  int i;
  //open the output file
  FILE* out = fopen(filename,"w");
	
  if(NULL == out){
    printf("Cannot open output file [%s]\n",filename);
    exit(1);
  }

  //this is the first regression
  LPLinklist p = linklist->Next;
  while(NULL!=p){
    //print the log marginal likelhood and the number of predictors
    //fprintf(out,"%.5lf\t%d",p->logmarglikA,p->lenA);
    fprintf(out, "index: %d\t--> Laplace approx.: %lf and Monte Carlo esti.: %lf, ", p->indexA, p->LlogmarglikA, p->MlogmarglikA);
    //now save the beta
    for(i=0;i<2;i++){
       if (i == 0){
           fprintf(out, "with beta_0:");
       }
       else if (i == 1){
           fprintf(out, ", beta_1: ");
       }
       fprintf(out,"\t%lf",gsl_matrix_get(p->beta, i, 0));
    }
    fprintf(out,"\n");

    //go to the next regression
    p = p->Next;
  }

  //close the output file
  fclose(out);

  return;
}

//this function deletes all the elements of the list
//with the head "regressions"
//remark that the head is not touched
void DeleteAllRegressions(LPLinklist linklist){
  //this is the first regression
  LPLinklist p = linklist->Next;
  LPLinklist pnext;

  while(NULL!=p){
    //save the link to the next element of p
    pnext = p->Next;

    //delete the element specified by p
    //first free the memory of the vector of regressors
    //delete[] p->A;
    gsl_matrix_free(p->beta);
    p->Next = NULL;
    delete p;

    //move to the next element
    p = pnext;
  }

  return;
}

