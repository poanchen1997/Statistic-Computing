rm(list = ls())
# setwd('C:/Users/user/Downloads/R/project/UW_homework/stat_534')

# install.packages("rcdd")
library(rcdd)

data = read.table("534binarydata.txt",header=FALSE)
# 1 -----------------------------------------------------------------------

#this is the version of the 'isValidLogistic' function
#based on Charles Geyers RCDD package
#returns TRUE if the calculated MLEs can be trusted
#returns FALSE otherwise
isValidLogisticRCDD <- function(response,explanatory,data)
{
  if(0==length(explanatory))
  {
    #we assume that the empty logistic regresion is valid
    return(TRUE); # if condition satisfied, the function will stop here and give answer
  }
  # suppressWarnings: not show the warnings
  logisticreg = suppressWarnings(glm(data[,response] ~ as.matrix(data[,as.numeric(explanatory)]),family=binomial(link=logit),x=TRUE));
  tanv = logisticreg$x;
  tanv[data[,response] == 1, ] <- (-tanv[data[,response] == 1, ]); # let the data that y actual value is one be minus?
  vrep = cbind(0, 0, tanv);  # add two column of 0 in front of tanv 
  #with exact arithmetic; takes a long time
  #lout = linearity(d2q(vrep), rep = "V");
  
  lout = linearity(vrep, rep = "V");
  return(length(lout)==nrow(data));
}

findNeighbor <- function(response, explanatory, data, subset_a){
  # find the neighbors
  neighbor_lst <- vector(mode = "list", length = length(explanatory))
  for (i in 1:length(explanatory)){
    test_sub <- subset_a
    if (explanatory[i] %in% test_sub){
      neighbor_lst[[i]] <- setdiff(test_sub, explanatory[i])
    } else {
      neighbor_lst[[i]] <- append(test_sub, explanatory[i])
    }
  }
  # eliminate the variables whose MLEs not exists
  for (j in 1:length(neighbor_lst)){
    if (isValidLogisticRCDD(response, neighbor_lst[[j]], data) == FALSE){
      #print("here we eliminate variables")
      neighbor_lst[j] <- list(NULL)
    } 
  }
  
  # neighbor list valid
  neighbor_lst_valid <- neighbor_lst[lengths(neighbor_lst) != 0]
  return(neighbor_lst_valid)
}

MC3search <- function(response, data, n_iter = 3){
  explanatory <- c(1:(response-1))
  
  # iteration 0
  k <- sample.int(length(explanatory), 1)
  subset_a <- c(sample(explanatory, k)) 
  while (isValidLogisticRCDD(response, subset_a, data) == FALSE){
    k <- sample.int(length(explanatory), 1)
    subset_a <- c(sample(explanatory, k))
  }
  subset_b <- subset_a
  cat("The start model is:          ", subset_b, "\n")
  
  for (i in 1:n_iter){
    cat("The", i, "iterations.\n")
    # iteration r
    neighbor_lst_valid <- findNeighbor(response, explanatory, data, subset_a)
    
    # sample from valid neighbor
    choose <- sample(length(neighbor_lst_valid), 1)
    neighbor_lst_valid_prime <- findNeighbor(response, explanatory, data, neighbor_lst_valid[[choose]])
    
    # step 5
    model_prime <- glm(data[,response] ~ as.matrix(data[,as.numeric(c(neighbor_lst_valid[[choose]]))]),family=binomial(link=logit))
    p_a_prime <- -model_prime$aic - log(length(neighbor_lst_valid_prime))
    
    # step 6
    model_r <- glm(data[,response] ~ as.matrix(data[,as.numeric(c(subset_a))]),family=binomial(link=logit))
    p_a_r <- -model_r$aic - log(length(neighbor_lst_valid))
    
    if (p_a_prime > p_a_r){
      #print("Falls into first condition")
      subset_a <- neighbor_lst_valid[[choose]]
      model_a <- glm(data[,response] ~ as.matrix(data[,as.numeric(c(subset_a))]),family=binomial(link=logit))
      model_b <- glm(data[,response] ~ as.matrix(data[,as.numeric(c(subset_b))]),family=binomial(link=logit))
      #Mbr_aic <- min(model_b$aic, model_a$aic)
      if (model_a$aic < model_b$aic){
        #print("Falls into first_first condition")
        subset_b <- subset_a
      } else {
        #print("Falls into first_second condition")
        subset_b <- subset_b
      } 
    } else if (p_a_prime <= p_a_r){
      #print("Falls into second condition")
      u <- runif(1, 0, 1) 
      if (log(u) < (p_a_prime - p_a_r)){
        #print("Falls into second_first condition")
        subset_a <- neighbor_lst_valid[[choose]]
        model_a <- glm(data[,response] ~ as.matrix(data[,as.numeric(c(subset_a))]),family=binomial(link=logit))
        model_b <- glm(data[,response] ~ as.matrix(data[,as.numeric(c(subset_b))]),family=binomial(link=logit))
        #Mbr_aic <- min(model_b$aic, model_a$aic)
        if (model_a$aic < model_b$aic){
          #print("Falls into second_first_first condition")
          subset_b <- subset_a
        } else {
          #print("Falls into second_first_second condition")
          subset_b <- subset_b
        } 
      }
    }
    cat("The iteration best model is: ", subset_b, "\n") # I print each iteration model to check my algorithm
  }
  
  best_model <- glm(data[, response] ~ as.matrix(data[, as.numeric(c(subset_b))]), family = binomial(link = logit))
  return(list(bestAICvars = subset_b, bestAIC = best_model$aic))
}



# s <- mc3(61, c(1:60), data, 3)
# cat("The AIC is: ", sort(s$model))
resp <- ncol(data)
for (a in 1:10){
  cat("The", a, "instances.\n")
  cat("=============================================")
  cat("\n")
  mResult <- MC3search(resp, data, 25)
  cat("\n")
  cat("The best model is: ", sort(mResult$bestAICvars))
  cat("\n")
  cat("The AIC is: ", mResult$bestAIC)
  cat("\n\n")
}
