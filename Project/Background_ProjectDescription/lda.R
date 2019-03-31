lda <-function(X1,X2) {

	
  #  Finds LDA parameters
  #  X1,X2 -> Class 1,2 samples (one data point per row)
  #  return values: a,b -> LDA parameters (a^T x + b = 0)
  #
  #  Author: Amin Zollanvari
  #  R version: 10/09/2015 

  X1 = as.matrix(X1)
  X2 = as.matrix(X2)
  
  m1 = colMeans(X1) # mean(X1)
  m2 = colMeans(X2) # mean(X2)
  
  n1 <- dim(X1)[1]
  n2 <- dim(X2)[1]
  S = ((n1-1)*cov(X1)+(n2-1)*cov(X2))/(n1+n2-2) # pooled covariance (reg)
  
  Si = solve(S);
  
  a = as.vector(Si%*%(m2-m1))
  b = as.numeric(0.5%*%t(m1-m2)%*%Si%*%(m1+m2))

  return(c(b,a))

}