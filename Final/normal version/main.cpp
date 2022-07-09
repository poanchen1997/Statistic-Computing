#include "final.h"

int main(){
    int n = 148;  // sample size
    int p = 61;   // number of variable, the last one is the outcome
    char datafilename[] = "534finalprojectdata.txt";  // input datafile
    char outputfile[] = "result.txt";  // output for test
    int i;

    //allocate the data matrix
    gsl_matrix* data = gsl_matrix_alloc(n, p);

    // read the data
    FILE* datafile = fopen(datafilename, "r");

    if(NULL==datafile){
        fprintf(stderr,"Cannot open data file [%s]\n",datafilename);
        return(0);
    }
    if(0!=gsl_matrix_fscanf(datafile,data)){ // record the data here
        fprintf(stderr,"File [%s] does not have the required format.\n",datafilename);
        return(0);
    }

    fclose(datafile);

    //create the head of the list of regressions
    LPLinklist linklist = new Linklist;
    //properly mark the end of the list
    linklist->Next = NULL;

    // test
    // random set-up
    const gsl_rng_type* T; // declare the variable we gonna use
    gsl_rng* r;
    gsl_rng_env_setup();   // initial the random number generator
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    
    gsl_matrix* beta_hat = gsl_matrix_alloc(2, 1);
    gsl_matrix* beta_bar = gsl_matrix_alloc(2, 1);
    gsl_matrix* x = gsl_matrix_alloc(n, 1);
    gsl_matrix* y = gsl_matrix_alloc(n, 1);

    for (i = 1; i < 61; i++){
        int A[] = {i};
        int B[] = {61};
        int len = 1;

        MakeSubmatrix_whole(data, n, A, len, x);
        MakeSubmatrix_whole(data, n, B, len, y);

        printf("Try the %d index of explanatory variables. --> \t", i);
        
        getcoefNR(y, x, beta_hat);
        
        double laplacevalue = getLaplaceApprox(y, x, beta_hat);
        //printf("The Laplace approximation: %lf.\n", laplacevalue);
    
        double montevalue = getMonteCarloIntegration(r, y, x);
        //printf("The Monte Carlo estimation: %lf.\n", montevalue);

        //printf("The beta_0 is %lf, beta_1 is %lf.\n", gsl_matrix_get(beta_hat, 0, 0), gsl_matrix_get(beta_hat, 1, 0));
        getPosteriorMeans(r, y, x, beta_hat, 10000, beta_bar);
        //printf("The beta_0 is %lf, beta_1 is %lf.\n", gsl_matrix_get(beta_bar, 0, 0), gsl_matrix_get(beta_bar, 1, 0));

        // insert the value in the linklist
        AddRegression(5, linklist, i, laplacevalue, montevalue, beta_bar);
    }

    // save the linklist to the output
    SaveRegressions(outputfile,linklist);

    //delete all regressions
    DeleteAllRegressions(linklist);
    
    // free r and matrix
    gsl_rng_free(r); 
    gsl_matrix_free(x);
    gsl_matrix_free(y);
    gsl_matrix_free(data);
    gsl_matrix_free(beta_hat);
    gsl_matrix_free(beta_bar);
    delete linklist; linklist = NULL;

    return(1);
}