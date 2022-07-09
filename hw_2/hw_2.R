rm(list = ls())
# setwd('C:/Users/user/Downloads/R/project/UW_homework/stat_534')

# 1 -----------------------------------------------------------------------

getLogisticAIC <- function(response,explanatory,data)
{
  #check if the regression has no explanatory variables
  if(0==length(explanatory))
  {
    #regression with no explanatory variables
    deviance = glm(data[,response] ~ 1,family=binomial(link=logit))$deviance;
  }
  else
  {
    #regression with at least one explanatory variable
    deviance = glm(data[,response] ~ as.matrix(data[,as.numeric(explanatory)]),
                   family=binomial(link=logit))$deviance;
  }
  return(deviance+2*(1+length(explanatory)));  # +1 for "beta_0"
}

# 2 -----------------------------------------------------------------------

forwardSearchAIC <- function(response,data,lastPredictor)
{
  
  #start with the empty regression with no predictors
  bestRegression = NULL;
  #calculate the AIC of the empty regression
  bestRegressionAIC = getLogisticAIC(response,bestRegression,data);
  cat('\n\n\n\nforwardSearch :: The empty logistic regression has AIC = ',bestRegressionAIC,'\n');

  #vector that keeps track of all the variables
  #that are currently NOT in the model
  VariablesNotInModel = 1:lastPredictor; # lastPredictor will be ncol(data) -1

  #add variables one at a time to the current regression
  #and retain that variable that gives the smallest values of AIC associated
  #Make the model that includes that variable to be the current model
  #if it gives a smaller AIC than the AIC of the current regression
  
  #stop when there are no variables that can be included in the model
  stepNumber = 0;
  stepRegression <- bestRegression
  stepRegressionAIC <- bestRegressionAIC

  while(length(VariablesNotInModel)>=1)
  {
    #record the number of steps performed
    stepNumber = stepNumber + 1;

    #create a vector that records the AIC values of the regressions
    #we are examining; the number of these regressions is equal
    #with the number of variables that are not in the model
    regAIC = vector('numeric',length(VariablesNotInModel));

    #take each variable that is not in the model
    #and include it in the model
    for (variable_index in 1:length(VariablesNotInModel)){
      testRegression <- stepRegression
      regAIC[variable_index] <- getLogisticAIC(response, c(append(testRegression, VariablesNotInModel[variable_index])), data)
    }
    if (min(regAIC) < stepRegressionAIC){
      optimal_for_add <- VariablesNotInModel[which.min(regAIC)]
      stepRegression <- append(stepRegression, optimal_for_add)
      VariablesNotInModel <- setdiff(VariablesNotInModel, optimal_for_add)
      stepRegressionAIC <- min(regAIC)
    } else {
      break
    }
  }

  bestRegression <- stepRegression
  bestRegressionAIC <- getLogisticAIC(response, bestRegression, data)
  
  return(list(aic=bestRegressionAIC,reg=bestRegression));
}

# 3 -----------------------------------------------------------------------
backwardSearchAIC <- function(response,data,lastPredictor)
{
  #start with the full regression that includes all the variables
  bestRegression = 1:lastPredictor;
  #calculate the AIC of the full regression
  bestRegressionAIC = getLogisticAIC(response,bestRegression,data);
  cat('\n\n\n\nbackwardSearch :: The full logistic regression has AIC = ',bestRegressionAIC,'\n');
  
  #sequentially delete one variable from the current regression
  #and retain that variable that gives the smallest AIC; make the model
  #in which that variable is deleted to be the current model if
  #this leads to a current model with a smaller AIC
  stepNumber = 0;
  stepRegression <- bestRegression
  stepRegressionAIC <- bestRegressionAIC
  
  while(length(stepRegression)>=1)
  {
    #record the number of steps performed
    stepNumber = stepNumber + 1;
    
    #create a vector that records the AIC values of the regressions
    #we are examining; the number of these regressions is equal
    #with the number of variables that are in the model
    regAIC = vector('numeric',length(stepRegression));
    
    for (variable_index in 1:length(stepRegression)){
      testRegression <- stepRegression
      regAIC[variable_index] <- getLogisticAIC(response, c(setdiff(testRegression, stepRegression[variable_index])), data)
    }
    if (min(regAIC) < stepRegressionAIC){
      optimal_for_del <- stepRegression[which.min(regAIC)]
      stepRegression <- setdiff(stepRegression, optimal_for_del)
      stepRegressionAIC <- min(regAIC)
    } else {
      break
    }
    bestRegression <- stepRegression
    bestRegressionAIC <- getLogisticAIC(response, bestRegression, data)
  }
  
  return(list(aic=bestRegressionAIC,reg=bestRegression));
}

# data = read.table("534binarydatasmall.txt",header=FALSE)
# data = read.table("534binarydata.txt",header=FALSE)
# response = ncol(data);
# lastPredictor = ncol(data)-1;
# nMaxPred = min(nMaxNumberExplanatory,lastPredictor)
# 
# forwardSearchAIC(response, data, lastPredictor)
# backwardSearchAIC(response, data, lastPredictor)

# 4 -----------------------------------------------------------------------
main <- function(datafile)
{
  #read the data
  data = read.table(datafile,header=FALSE);
  
  #the sample size is 148 (number of rows)
  #the explanatory variables are the first 60 columns
  #the last column is the binary response
  response = ncol(data);
  lastPredictor = ncol(data)-1;
  
  #perform a forward "greedy" search for the best logistic regression
  #i.e., the logistic regression with the smallest AIC
  forwardResults = forwardSearchAIC(response,data,lastPredictor);
  
  #perform a backward "greedy" search for the best logistic regression
  backwardResults = backwardSearchAIC(response,data,lastPredictor);
  
  #output the results of our searches
  cat('\n\nForward search gives regression with ',length(forwardResults$reg),'explanatory variables [');
  if(length(forwardResults$reg)>=1)
  {
    for(i in 1:length(forwardResults$reg)) cat(' ',forwardResults$reg[i]);
  }
  cat('] with AIC = ',forwardResults$aic,'\n');
  
  cat('\n\nBackward search gives regression with ',length(backwardResults$reg),'explanatory variables [');
  if(length(backwardResults$reg)>=1)
  {
    for(i in 1:length(backwardResults$reg)) cat(' ',backwardResults$reg[i]);
  }
  cat('] with AIC = ',backwardResults$aic,'\n');
}

main('534binarydata.txt');

# ctrl + shift + c
# The regression model my algorithm suggest me is different between them, however,
# they have the same AIC. As for the BIC part, the algorithm suggest us the same 
# regression while the BIC is not the same.

#================================================================
# BIC part
getLogisticBIC <- function(response,explanatory,data)
{
  #check if the regression has no explanatory variables
  if(0==length(explanatory))
  {
    #regression with no explanatory variables
    deviance = glm(data[,response] ~ 1,family=binomial(link=logit))$deviance;
  }
  else
  {
    #regression with at least one explanatory variable
    deviance = glm(data[,response] ~ as.matrix(data[,as.numeric(explanatory)]),family=binomial(link=logit))$deviance;
  }
  return(deviance+log(nrow(data))*(1+length(explanatory)));
}
forwardSearchBIC <- function(response,data,lastPredictor)
{
  
  #start with the empty regression with no predictors
  bestRegression = NULL;
  #calculate the BIC of the empty regression
  bestRegressionBIC = getLogisticBIC(response,bestRegression,data);
  cat('\n\n\n\nforwardSearch :: The empty logistic regression has BIC = ',bestRegressionBIC,'\n');
  
  #vector that keeps track of all the variables
  #that are currently NOT in the model
  VariablesNotInModel = 1:lastPredictor; # lastPredictor will be ncol(data) -1
  
  #add variables one at a time to the current regression
  #and retain that variable that gives the smallest values of BIC associated
  #Make the model that includes that variable to be the current model
  #if it gives a smaller BIC than the BIC of the current regression
  
  #stop when there are no variables that can be included in the model
  stepNumber = 0;
  stepRegression <- bestRegression
  stepRegressionBIC <- bestRegressionBIC
  
  while(length(VariablesNotInModel)>=1)
  {
    #record the number of steps performed
    stepNumber = stepNumber + 1;
    
    #create a vector that records the BIC values of the regressions
    #we are examining; the number of these regressions is equal
    #with the number of variables that are not in the model
    regBIC = vector('numeric',length(VariablesNotInModel));
    
    #take each variable that is not in the model
    #and include it in the model
    for (variable_index in 1:length(VariablesNotInModel)){
      testRegression <- stepRegression
      regBIC[variable_index] <- getLogisticBIC(response, c(append(testRegression, VariablesNotInModel[variable_index])), data)
    }
    if (min(regBIC) < stepRegressionBIC){
      optimal_for_add <- VariablesNotInModel[which.min(regBIC)]
      stepRegression <- append(stepRegression, optimal_for_add)
      VariablesNotInModel <- setdiff(VariablesNotInModel, optimal_for_add)
      stepRegressionBIC <- min(regBIC)
    } else {
      break
    }
  }
  
  bestRegression <- stepRegression
  bestRegressionBIC <- getLogisticBIC(response, bestRegression, data)
  
  return(list(bic=bestRegressionBIC,reg=bestRegression));
}

backwardSearchBIC <- function(response,data,lastPredictor)
{
  #start with the full regression that includes all the variables
  bestRegression = 1:lastPredictor;
  #calculate the BIC of the full regression
  bestRegressionBIC = getLogisticBIC(response,bestRegression,data);
  cat('\n\n\n\nbackwardSearch :: The full logistic regression has BIC = ',bestRegressionBIC,'\n');
  
  #sequentially delete one variable from the current regression
  #and retain that variable that gives the smallest BIC; make the model
  #in which that variable is deleted to be the current model if
  #this leads to a current model with a smaller BIC
  stepNumber = 0;
  stepRegression <- bestRegression
  stepRegressionBIC <- bestRegressionBIC
  
  while(length(stepRegression)>=1)
  {
    #record the number of steps performed
    stepNumber = stepNumber + 1;
    
    #create a vector that records the BIC values of the regressions
    #we are examining; the number of these regressions is equal
    #with the number of variables that are in the model
    regBIC = vector('numeric',length(stepRegression));
    
    for (variable_index in 1:length(stepRegression)){
      testRegression <- stepRegression
      regBIC[variable_index] <- getLogisticBIC(response, c(setdiff(testRegression, stepRegression[variable_index])), data)
    }
    if (min(regBIC) < stepRegressionBIC){
      optimal_for_del <- stepRegression[which.min(regBIC)]
      stepRegression <- setdiff(stepRegression, optimal_for_del)
      #VariablesNotInModel <- setdiff(VariablesNotInModel, optimal_for_add)
      stepRegressionBIC <- min(regBIC)
    } else {
      break
    }
    bestRegression <- stepRegression
    bestRegressionBIC <- getLogisticBIC(response, bestRegression, data)
  }
  
  return(list(bic=bestRegressionBIC,reg=bestRegression));
}

main <- function(datafile)
{
  #read the data
  data = read.table(datafile,header=FALSE);
  
  #the sample size is 148 (number of rows)
  #the explanatory variables are the first 60 columns
  #the last column is the binary response
  response = ncol(data);
  lastPredictor = ncol(data)-1;
  
  #perform a forward "greedy" search for the best logistic regression
  #i.e., the logistic regression with the smallest BIC
  forwardResults = forwardSearchBIC(response,data,lastPredictor);
  
  #perform a backward "greedy" search for the best logistic regression
  backwardResults = backwardSearchBIC(response,data,lastPredictor);
  
  #output the results of our searches
  cat('\n\nForward search gives regression with ',length(forwardResults$reg),'explanatory variables [');
  if(length(forwardResults$reg)>=1)
  {
    for(i in 1:length(forwardResults$reg)) cat(' ',forwardResults$reg[i]);
  }
  cat('] with BIC = ',forwardResults$bic,'\n');
  
  cat('\n\nBackward search gives regression with ',length(backwardResults$reg),'explanatory variables [');
  if(length(backwardResults$reg)>=1)
  {
    for(i in 1:length(backwardResults$reg)) cat(' ',backwardResults$reg[i]);
  }
  cat('] with BIC = ',backwardResults$bic,'\n');
}

main('534binarydata.txt');


  
  
  
  
  
  
  
