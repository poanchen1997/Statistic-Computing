inverseLogit <- function(x)
{
  return(exp(x)/(1+exp(x))); 
}

inverseLogit2 <- function(x)
{
  return(exp(x)/(1+exp(x))^2); 
}

getPi <- function(x,beta)
{
  x0 = cbind(rep(1,length(x)),x);  # beta_0 = 1, beta_1 = data[, explanatory]
  return(inverseLogit(x0%*%beta));
}

getPi2 <- function(x,beta)
{
  x0 = cbind(rep(1,length(x)),x);
  return(inverseLogit2(x0%*%beta));   # not have minus symbol here, put it in the last part of hessian
}

logisticLoglik <- function(y,x,beta)
{
  Pi = getPi(x,beta);
  return(sum(y*log(Pi))+sum((1-y)*log(1-Pi)));
}

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

getPD <- function(response, explanatory, data){
  y = data[,response];
  x = data[,explanatory];
  beta_star <- getcoefNR_forlstar(response, explanatory, data)
  return(log(2*pi)+ l_star(y, x, beta_star) - 0.5*log(det(-getHessian_forlstar(y, x, beta_star))))
}

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