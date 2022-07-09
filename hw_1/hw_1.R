rm(list = ls())
setwd('C:/Users/user/Downloads/R/project/UW_homework/stat_534')
# 1 -----------------------------------------------------------------------

logdet <- function(R){
  log(prod(eigen(R)$values))
}


R <- matrix(c(13, -4, 2, -4, 11, -2, 2, -2, 8), 3, 3, byrow=TRUE)
R
logdet(R)

# check 
testlogdet <- function(R){
  log(det(R))
}

testlogdet(R)
# 2 -----------------------------------------------------------------------
library(matlib)
data <- read.table('erdata.txt', head = FALSE)
A <- c(2, 5, 10)

logmarglik <- function(data, A){
  # A is a vector
  M_A <- diag(length(A)) + t(data[A])%*%as.matrix(data[A])
  ans <- lgamma((nrow(data) + length(A) + 2)/2) -
         lgamma((length(A) + 2)/2) -
         0.5*logdet(M_A) - 
         ((nrow(data) + length(A) + 2)/2)*log(1 + t(data[1])%*%as.matrix(data[1]) - t(data[1])%*%as.matrix(data[A])%*%inv(M_A)%*%t(data[A])%*%as.matrix(data[1]))
  return(ans) 
}

logmarglik(data, A)



# check
testlogmarglik <- function(data, A){
  # A is a vector
  M_A <- diag(length(A)) + t(data[A])%*%as.matrix(data[A])
  log(gamma((nrow(data)+length(A) + 2)/2)/gamma((length(A) + 2)/2)*
        det(M_A)^(-0.5)*
        (1 + t(data[1])%*%as.matrix(data[1]) - t(data[1])%*%as.matrix(data[A])%*%inv(M_A)%*%t(data[A])%*%as.matrix(data[1]))^(-(nrow(data)+length(A)+2)/2))
}

testlogmarglik(data, A)
