/*
    run the program by:
    mpirun -np 10 final
*/

#include "final.h"

// For MPI communication
#define GETALLRESULT 1
#define SHUTDOWNTAG	0

// Used to determine PRIMARY or REPLICA
static int myrank;

// Global variables
int nobservations = 148;
int nvariables = 61;


int main(int argc, char* argv[]){
    char datafilename[] = "534finalprojectdata.txt";  // input datafile

    // start the MPI session
    MPI_Init(&argc, &argv);

    // the ID fot the process
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    //allocate the data matrix
    gsl_matrix* data = gsl_matrix_alloc(nobservations, nvariables);

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

    // random set-up
    const gsl_rng_type* T; // declare the variable we gonna use
    gsl_rng* r;
    gsl_rng_env_setup();   // initial the random number generator
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    if (myrank == 0){
        primary();
    } 
    else{
        replica(myrank, data, r);
    }

    // free the matrix
    gsl_rng_free(r); 
    gsl_matrix_free(data);

    //FINALIZE THE MPI SESSION
    MPI_Finalize();

    return 0;
}

void primary(){
    int var, rank, ntasks, jobsRunning;
    int work[1];
    double workresults[5];
    char outputlist[] = "resultlist.txt";
    //FILE* fout;
    MPI_Status status;
    gsl_matrix* beta = gsl_matrix_alloc(2, 1);

    //create the head of the list of regressions
    LPLinklist linklist = new Linklist;
    //properly mark the end of the list
    linklist->Next = NULL;

    // Find out how many replicas there are
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    fprintf(stdout, "Total Number of processors = %d.\n", ntasks);

    // now loop the all the variables to compute the value
    jobsRunning = 1;

    for (var = 1; var < nvariables; var++){
        // tell the replica which variable to compute
        work[0] = var;

        if (jobsRunning < ntasks){ // check if we have enough available processor
            MPI_Send(&work, 1, MPI_INT, jobsRunning, GETALLRESULT, MPI_COMM_WORLD);
            printf("Primary sends out work request [%d] to replica [%d].\n", work[0], jobsRunning);
            jobsRunning++;
        }
        else{ // if all the processors are in use
            MPI_Recv(workresults, 5, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            printf("Primary has received the result of work request [%d] from replica [%d].\n", (int)workresults[0], status.MPI_SOURCE);

            // print out the result
            gsl_matrix_set(beta, 0, 0, workresults[3]);
            gsl_matrix_set(beta, 1, 0, workresults[4]);
            AddRegression(5, linklist, workresults[0], workresults[1], workresults[2], beta);

            // tell the replica to do the next work
            printf("Primary sends out work request [%d] to replica [%d].\n", work[0], status.MPI_SOURCE);
            MPI_Send(&work, 1, MPI_INT, status.MPI_SOURCE, GETALLRESULT, MPI_COMM_WORLD);
        }
    }

    for (rank = 1; rank < jobsRunning; rank++){
        MPI_Recv(workresults, 5, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("Primary has received the result of work request [%d].\n", (int)workresults[0]);

        // save the results received
        gsl_matrix_set(beta, 0, 0, workresults[3]);
        gsl_matrix_set(beta, 1, 0, workresults[4]);
        AddRegression(5, linklist, workresults[0], workresults[1], workresults[2], beta);
    }

    // shut down the replica processes
    printf("====Tell the replicas to shutdown.====\n");
    for (rank = 1; rank < ntasks; rank++){
        printf("Primary is shutting down replica [%d].\n", rank);
        MPI_Send(0, 0, MPI_INT, rank, SHUTDOWNTAG, MPI_COMM_WORLD);
    }

    printf("We're now to the end of Primary code.\n");

    // save the linklist
    SaveRegressions(outputlist, linklist);

    // delete all regressions
    DeleteAllRegressions(linklist);

    // free matrix
    gsl_matrix_free(beta);
    delete linklist; linklist = NULL;

    return;
}

void replica(int replicaname, gsl_matrix* data, gsl_rng* r){
    int work[1];
    double workresults[5];
    MPI_Status status;
    int notDone = 1;
    gsl_matrix* beta_hat = gsl_matrix_alloc(2, 1);
    gsl_matrix* beta_bar = gsl_matrix_alloc(2, 1);
    gsl_matrix* x = gsl_matrix_alloc(nobservations, 1);
    gsl_matrix* y = gsl_matrix_alloc(nobservations, 1);
    int A[1], B[] = {61}, len = 1;
    double ans = 0.0;
    double laplacevalue, montevalue;

    while (notDone){
        printf("Replica %d is waiting.\n", replicaname);
        MPI_Recv(&work, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // switch on the type of work request
        switch(status.MPI_TAG){
            case GETALLRESULT:
                printf("Replica %d has received work request [%d].\n", replicaname, work[0]);
                // generate y and x for the below computing
                A[0] = work[0];
                MakeSubmatrix_whole(data, nobservations, A, len, x);
                MakeSubmatrix_whole(data, nobservations, B, len, y);

                // get beta_hat by "getcoefNR"
                getcoefNR(y, x, beta_hat);

                // compute Laplace approximation
                laplacevalue = getLaplaceApprox(y, x, beta_hat);
                workresults[1] = laplacevalue;

                // compute Monte Carlo estimation
                montevalue = getMonteCarloIntegration(r, y, x);
                workresults[2] = montevalue;

                // get beta_bar by "getPosterMeans"
                getPosteriorMeans(r, y, x, beta_hat, 10000, beta_bar);
                workresults[3] = gsl_matrix_get(beta_bar, 0, 0);
                workresults[4] = gsl_matrix_get(beta_bar, 1, 0);

                // get the variable index of the regression
                workresults[0] = (double)work[0];

                // send the results
                MPI_Send(&workresults, 5, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                printf("Replica %d finished processing work request [%d].\n", replicaname, work[0]);
                break;
            case SHUTDOWNTAG:
                printf("Replica %d was told to shutdown.\n", replicaname);
                return;
            default:
                notDone = 0;
                printf("The replica code should never get here.\n");
                return;
        }
    }
    
    // free matrix
    gsl_matrix_free(beta_hat);
    gsl_matrix_free(beta_bar);
    gsl_matrix_free(x);
    gsl_matrix_free(y);

    return;
}

