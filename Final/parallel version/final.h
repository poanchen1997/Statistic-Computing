#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <iomanip>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_linalg.h>

// the link list to store 
typedef struct myLinklist* LPLinklist;
typedef struct myLinklist Linklist;

struct myLinklist
{
  int indexA;            // index number of regressors
  double LlogmarglikA;   // log marginal likelihood of the regression --> Laplace approximation
  double MlogmarglikA;   // log marginal likelihood of the regression --> Monte Carlo integration
  gsl_matrix* beta;      // The beta esmation from posterior distribution

  LPLinklist Next; //link to the next regression
};

// declare function:
void printmatrix(char* filename,gsl_matrix* m);
double inverseLogit(double x);
double inverseLogit2(double x);
void getPi(gsl_matrix* x, gsl_matrix* beta, gsl_matrix* pi);
void getPi2(gsl_matrix* x, gsl_matrix* beta, gsl_matrix* pi2);
void MakeSubmatrix_whole(gsl_matrix* M,
			  int lenIndRow,
			  int* IndColumn,int lenIndColumn, gsl_matrix* subM);
void inverse(gsl_matrix* K, gsl_matrix* inverse);
double logisticLoglik(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta);
double lStar(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta);
void getGradient(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, gsl_matrix* gradient);
void getHessian(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, gsl_matrix* Hessian);
void getcoefNR(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta);
void makeCholesky(gsl_matrix* K, gsl_matrix* out);
void mvrnorm(int sampletimes, gsl_rng* mystream, gsl_matrix* beta, gsl_matrix* sigma, gsl_matrix* sample);
void transposematrix(gsl_matrix* m, gsl_matrix* tm);
void mhLogisticRegression(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, gsl_matrix* invNegHessian, gsl_matrix* beta_output);
double getLogDeterminant(gsl_matrix* inputMatrix, int n, int p);
void MakeSubmatrix(gsl_matrix* M,
			  int* IndRow,int lenIndRow,
			  int* IndColumn,int lenIndColumn, gsl_matrix* subM);
double getLaplaceApprox(gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode);
void getPosteriorMeans(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode, int NumberOfIteration, gsl_matrix* betaBayes);
double getMonteCarloIntegration(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x);
void AddRegression(int nMax_in_list, LPLinklist linklist, int indexA, 
                    double LlogmarglikA, double MlogmarglikA, gsl_matrix* beta);
int getCount(LPLinklist linklist);
void DeleteLastRegression(LPLinklist linklist);
void SaveRegressions(char* filename,LPLinklist linklist);
void DeleteAllRegressions(LPLinklist linklist);
void primary();
void replica(int replicname, gsl_matrix* data, gsl_rng* r);

