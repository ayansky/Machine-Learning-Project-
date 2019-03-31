/*
 * errorEstimation.h
 *
 */

#ifndef ERRORESTIMATION_H_
#define ERRORESTIMATION_H_

#include "standardHeaders.h"
#include "utilities.h"
#include "featureSelection.h"
#include "classifiers.h"
#include "random.h"
#include "main.h"

double ldaBolster(double** X, int* y, int N, int d, int* ind, LDA lda);
double lsvmBolster(double** X, int *y, int N, int d, int* ind, int N_bolster, long* seed);
double ksvmBolster(double** X, int *y, int N, int d, int* ind, int N_bolster, long* seed);

double ldaLOO(double** X, int* y, int N, int D, int d);
double lsvmLOO(double** X, int* y, int N, int D, int d);
double ksvmLOO(double** X, int* y, int N, int D, int d);

double ldaCVkFold(double** X, int* y, int N, int D, int d, int K, int R, long* seed);
double lsvmCVkFold(double** X, int* y, int N, int D, int d, int K, int R, long* seed);
double ksvmCVkFold(double** X, int* y, int N, int D, int d, int K, int R, long* seed);

double ldaBoot632(double** X, int* y, int N, int D, int d, int B, double resub, long* seed);
double lsvmBoot632(double** X, int* y, int N, int D, int d, int B, double resub, long* seed);
double ksvmBoot632(double** X, int* y, int N, int D, int d, int B, double resub, long* seed);
#endif /*ERRORESTIMATION_H_*/
