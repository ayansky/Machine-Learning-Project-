resub.lda <- function(X0,X1,a,m)
{
  #  Computes resubstitution error for LDA
  #  X0,X1 -> Class 0,1 samples (one data point per row)
  #  a,m -> LDA parameters (a^T x + m = 0)
  #  return values: e1,e2,e -> error rates (prim,sec,overall)
  #  Author: Amin Zollanvari
  #  R version: 10/09/2015 

  X0 = as.matrix(X0)
  X1 = as.matrix(X1)

  n0 <- dim(X0)[1]
  n1 <- dim(X1)[1]

  E0 = X0%*%a + m
  E1 = X1%*%a + m

  e0 = sum(E0>0);
  e1 = sum(E1<=0)

  return(list(e=(e0+e1)/(n0+n1),e0 = e0/n0,e1 = e1/n1))

}