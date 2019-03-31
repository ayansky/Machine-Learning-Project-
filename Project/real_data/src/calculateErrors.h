/*
 * calculateErrors.h
 *
 */

#ifndef CALCULATEERRORS_H_
#define CALCULATEERRORS_H_

#include "standardHeaders.h"
#include "print.h"
#include "utilities.h"
#include "matrixOperations.h"
#include "dataGeneration.h"
#include "featureSelection.h"
#include "errorEstimation.h"
#include "classifiers.h"


struct AllErrors {
	double lda_true_error;
	double lsvm_true_error;
	double ksvm_true_error;

	double lda_resub_error;
	double lsvm_resub_error;
	double ksvm_resub_error;

	double lda_bolster_error;
	double lsvm_bolster_error;
	double ksvm_bolster_error;

	double lda_loo_error;
	double lsvm_loo_error;
	double ksvm_loo_error;

	double lda_cvkfold_error;
	double lsvm_cvkfold_error;
	double ksvm_cvkfold_error;

	double lda_boot632_error;
	double lsvm_boot632_error;
	double ksvm_boot632_error;
};

void calculateErrors(
		double** X,
		int* y,
		int N_trn,
		int N_tst,
		int D,
		int d,
		long* seed,
		AllErrors* all_errors);

#endif /* CALCULATEERRORS_H_ */
