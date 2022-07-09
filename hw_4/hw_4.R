rm(list = ls())
library(snow)
library(MASS)

#setwd('C:/Users/user/Downloads/R/project/UW_homework/stat_534')
#data = read.table("534binarydata.txt",header=FALSE);

## function from lecture
getcoefglm <- function(response,explanatory,data)
{
  return(coef(glm(data[,response] ~ data[,explanatory],family=binomial(link=logit))));
}

#the inverse of the logit function
inverseLogit <- function(x)
{
  return(exp(x)/(1+exp(x))); 
}

#function for the computation of the Hessian
inverseLogit2 <- function(x)
{
  return(exp(x)/(1+exp(x))^2); 
}

#computes pi_i = P(y_i = 1 | x_i)
getPi <- function(x,beta)
{
  x0 = cbind(rep(1,length(x)),x);  # beta_0 = 1, beta_1 = data[, explanatory]
  return(inverseLogit(x0%*%beta));
}

#another function for the computation of the Hessian
getPi2 <- function(x,beta)
{
  x0 = cbind(rep(1,length(x)),x);
  return(inverseLogit2(x0%*%beta));   # not have minus symbol here, put it in the last part of hessian
}

#logistic log-likelihood (formula (3) in your handout)
logisticLoglik <- function(y,x,beta)
{
  Pi = getPi(x,beta);
  return(sum(y*log(Pi))+sum((1-y)*log(1-Pi)));
}

#obtain the gradient for Newton-Raphson
getGradient <- function(y,x,beta)
{
  gradient = matrix(0,2,1);
  Pi = getPi(x,beta);
  
  gradient[1,1] = sum(y-Pi);
  gradient[2,1] = sum((y-Pi)*x);
  
  return(gradient);
}

#obtain the Hessian for Newton-Raphson
getHessian <- function(y,x,beta)
{
  hessian = matrix(0,2,2);
  Pi2 = getPi2(x,beta);
  
  hessian[1,1] = sum(Pi2);
  hessian[1,2] = sum(Pi2*x);
  hessian[2,1] = hessian[1,2];
  hessian[2,2] = sum(Pi2*x^2);
  
  return(-hessian);
}

#this function implements our own Newton-Raphson procedure
getcoefNR <- function(response,explanatory,data)
{
  #2x1 matrix of coefficients`
  beta = matrix(0,2,1);  # start of the guess
  y = data[,response];
  x = data[,explanatory];
  
  #current value of log-likelihood
  currentLoglik = logisticLoglik(y,x,beta);
  
  #infinite loop unless we stop it someplace inside
  while(1)
  {
    newBeta = beta - solve(getHessian(y,x,beta))%*%getGradient(y,x,beta);
    # the function to find the inverse function, same as "inv"
    newLoglik = logisticLoglik(y,x,newBeta);
    
    #at each iteration the log-likelihood must increase
    if(newLoglik<currentLoglik)
    {
      cat("CODING ERROR!!\n");
      break;
    }
    
    beta = newBeta;
    #stop if the log-likelihood does not improve by too much
    if(newLoglik-currentLoglik<1e-6)
    {
      break; 
    }
    currentLoglik = newLoglik;
  }
  
  return(beta);
}
# 1 -----------------------------------------------------------------------
l_star <- function(y, x, beta){
  return(-log(2*pi) - 0.5*(beta[1]^2 + beta[2]^2) + logisticLoglik(y, x, beta))
}

getGradient_forlstar <- function(y, x, beta){
  gradient = matrix(0,2,1);
  Pi = getPi(x,beta);
  
  gradient[1,1] = -beta[1] + sum(y-Pi);
  gradient[2,1] = -beta[2] + sum((y-Pi)*x);
  
  return(gradient);
}

getHessian_forlstar <- function(y,x,beta)
{
  hessian = matrix(0,2,2);
  Pi2 = getPi2(x,beta);
  # put negative sign in the last step
  hessian[1,1] = 1 + sum(Pi2);
  hessian[1,2] = sum(Pi2*x);
  hessian[2,1] = hessian[1,2];
  hessian[2,2] = 1 + sum(Pi2*x^2);
  
  return(-hessian);  
}

getcoefNR_forlstar <- function(response,explanatory,data)
{
  #2x1 matrix of coefficients`
  beta = matrix(0,2,1);  # start of the guess
  y = data[,response];
  x = data[,explanatory];
  
  #current value of log-likelihood
  currentLoglik = logisticLoglik(y,x,beta);
  
  #infinite loop unless we stop it someplace inside
  while(1)
  {
    newBeta = beta - solve(getHessian_forlstar(y,x,beta))%*%getGradient_forlstar(y,x,beta);
    # the function to find the inverse function, same as "inv"
    newLoglik = logisticLoglik(y,x,newBeta);
    
    #at each iteration the log-likelihood must increase
    if(newLoglik<currentLoglik)
    {
      cat("CODING ERROR!!\n");
      break;
    }
    
    beta = newBeta;
    #stop if the log-likelihood does not improve by too much
    if(newLoglik-currentLoglik<1e-6)
    {
      break; 
    }
    currentLoglik = newLoglik;
  }
  
  return(beta);
}

# getPD <- function(y, x, data){
#   beta_star <- getcoefNR_forlstar(y, x, data)
#   return(2*pi*exp(l_star(y, x, beta_star))*(det(-getHessian_forlstar(y, x, beta_star)))^(-0.5))
# }

# version for computing log(PD)
getPD <- function(response, explanatory, data){
  y = data[,response];
  x = data[,explanatory];
  beta_star <- getcoefNR_forlstar(response, explanatory, data)
  return(log(2*pi)+ l_star(y, x, beta_star) - 0.5*log(det(-getHessian_forlstar(y, x, beta_star))))
}



# test
# response = 61;
# explanatory = 25;
# getcoefNR(response, explanatory, data)
# beta <- getcoefglm(response, explanatory, data)
# beta
# l_star(response, explanatory, beta)
# getGradient_forlstar(response, explanatory, beta)
# getHessian_forlstar(response, explanatory, beta)
# beta <- getcoefNR_forlstar(response, explanatory, data)
# getPD(response, explanatory, data)

# 2 -----------------------------------------------------------------------
MH_AL <- function(response, explanatory, data, iteration){
  y = data[,response]
  x = data[,explanatory]

  # calculate the start beta
  startbeta <- getcoefNR_forlstar(response, explanatory, data)
  beta_previous <- startbeta
  beta0_vector <- rep(NA, iteration)
  beta1_vector <- rep(NA, iteration)
  beta0_vector[1] <- beta_previous[1]
  beta1_vector[1] <- beta_previous[2]
  
  for (i in 2:iteration){
    # sample from the multivariate normal distribution
    sample <- mvrnorm(mu = beta_previous, Sigma = -solve(getHessian_forlstar(y, x, beta_previous)))
    
    # sample a uniform to check if we gonna change the current state
    u <- runif(1, 0, 1)

    # check if we stay in the current state
    if (log(u) <= (l_star(y, x, sample) - l_star(y, x, beta_previous))){
      beta_previous <- sample
    } # else the beta_previous don't change
    
    # record the beta for each iterations
    beta0_vector[i] <- beta_previous[1]
    beta1_vector[i] <- beta_previous[2]
  }
  
  return(list(beta0 = beta0_vector, beta1 = beta1_vector))
}
# result <- MH_AL(61, 25, data, 10000)
# 
# beta0_bar <- sum(result$beta0)/10000
# beta1_bar <- sum(result$beta1)/10000

# 3 -----------------------------------------------------------------------
bayesLogistic = function(apredictor,response,data,NumberOfIterations)
{
  library(snow)
  library(MASS)
  # import the function we build above
  source('function_for_hw4.R')
  
  # calculate laplace approximation
  laplace <- getPD(response, apredictor, data)
  beta_bayes <- getcoefNR_forlstar(response, apredictor, data)
  
  # the approximation computing by sample mean from Metropolis-Hasting Algorithm
  beta_mle <- MH_AL(response, apredictor, data, NumberOfIterations)
  
  return(list(apredictor = apredictor, logmarglik = laplace, 
              beta0bayes = beta_bayes[1], beta1bayes = beta_bayes[2], 
              beta0mle = sum(beta_mle$beta0)/10000, beta1mle = sum(beta_mle$beta1)/10000))
}

#bayesLogistic(25, 61, data, 10000)

#PARALLEL VERSION
#datafile = the name of the file with the data
#NumberOfIterations = number of iterations of the Metropolis-Hastings algorithm
#clusterSize = number of separate processes; each process performs one or more
#univariate regressions
main <- function(datafile,NumberOfIterations,clusterSize)
{
  #read the data
  data = read.table(datafile,header=FALSE);
  
  #the sample size is 148 (number of rows)
  #the explanatory variables are the first 60 columns for '534binarydata.txt'
  #the last column is the binary response
  response = ncol(data);
  lastPredictor = ncol(data)-1;
  
  #initialize a cluster for parallel computing
  cluster <- makeCluster(clusterSize, type = "SOCK")
  
  #run the MC3 algorithm from several times
  results = clusterApply(cluster, 1:lastPredictor, bayesLogistic,
                         response,data,NumberOfIterations);
  
  #print out the results
  for(i in 1:lastPredictor)
  {
    cat('Regression of Y on explanatory variable ',results[[i]]$apredictor,
        ' has log marginal likelihood ',results[[i]]$logmarglik,
        ' with beta0 = ',results[[i]]$beta0bayes,' (',results[[i]]$beta0mle,')',
        ' and beta1 = ',results[[i]]$beta1bayes,' (',results[[i]]$beta1mle,')',
        '\n');    
  }
  
  #destroy the cluster
  stopCluster(cluster);  
}

#NOTE: YOU NEED THE PACKAGE 'SNOW' FOR PARALLEL COMPUTING
#require(snow);

#this is where the program starts
main('534binarydata.txt',10000,10);
