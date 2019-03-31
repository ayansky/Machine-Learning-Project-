/*
 * calculateErrors.cpp
 *
 */

#include "calculateErrors.h"

void calculateErrors(
		double** X,
		int* y,
		int N_trn,
		int N_tst,
		int D,
		int d,
		long* seed,
		AllErrors* all_errors)
{
	double dinit = 0.00;

	int N_bolster = 25; // number of bolster sample point
	int K = 5; // k-fold CV
	int R = 5; // number of iterations in CV
	int B = 50; // number of bootstrap replicates

	SimulationData data_trn;
	SimulationData data_tst;

	int* best_features;

	LDA model_LDA;

	svm_model *model_LSVM;	//svm training model
	svm_node *subdata_LSVM;	//svm training data
	svm_problem subcl_LSVM;	//svm training data structure

	svm_model *model_KSVM;	//svm training model
	svm_node *subdata_KSVM;	//svm training data
	svm_problem subcl_KSVM;	//svm training data structure

	(*all_errors).lda_true_error = 0.00;
	(*all_errors).lsvm_true_error = 0.00;
	(*all_errors).ksvm_true_error = 0.00;

	(*all_errors).lda_resub_error = 0.00;
	(*all_errors).lsvm_resub_error = 0.00;
	(*all_errors).ksvm_resub_error = 0.00;

	(*all_errors).lda_bolster_error = 0.00;
	(*all_errors).lsvm_bolster_error = 0.00;
	(*all_errors).ksvm_bolster_error = 0.00;

	(*all_errors).lda_loo_error = 0.00;
	(*all_errors).lsvm_loo_error = 0.00;
	(*all_errors).ksvm_loo_error = 0.00;

	(*all_errors).lda_cvkfold_error = 0.00;
	(*all_errors).lsvm_cvkfold_error = 0.00;
	(*all_errors).ksvm_cvkfold_error = 0.00;

	data_trn.data = make_2D_matrix(N_trn, D, dinit);
	data_trn.labels = new int [N_trn];
	data_tst.data = make_2D_matrix(N_tst, D, dinit);
	data_tst.labels = new int [N_tst];

	dataGeneration(X, y, N_trn, N_tst, D, seed, &data_trn, &data_tst);

	best_features = new int [d];
	featureSelection(data_trn.data, data_trn.labels, data_trn.N, data_trn.D, d, best_features);

	model_LDA.a = new double [d];
	ldaTrn(data_trn.data, data_trn.labels, data_trn.N, d, best_features, &model_LDA);

	model_LSVM = svmTrn(data_trn.data, data_trn.labels, data_trn.N, d, best_features, 0, &subdata_LSVM, &subcl_LSVM);
	model_KSVM = svmTrn(data_trn.data, data_trn.labels, data_trn.N, d, best_features, 2, &subdata_KSVM, &subcl_KSVM);

	(*all_errors).lda_true_error = ldaTst(data_tst.data, data_tst.labels, data_tst.N, d, best_features, model_LDA);
	(*all_errors).lsvm_true_error = svmTst(data_tst.data, data_tst.labels, data_tst.N, d, best_features, model_LSVM);
	(*all_errors).ksvm_true_error = svmTst(data_tst.data, data_tst.labels, data_tst.N, d, best_features, model_KSVM);

	(*all_errors).lda_resub_error = ldaTst(data_trn.data, data_trn.labels, data_trn.N, d, best_features, model_LDA);
	(*all_errors).lsvm_resub_error = svmTst(data_trn.data, data_trn.labels, data_trn.N, d, best_features, model_LSVM);
	(*all_errors).ksvm_resub_error = svmTst(data_trn.data, data_trn.labels, data_trn.N, d, best_features, model_KSVM);

	(*all_errors).lda_bolster_error = ldaBolster(data_trn.data, data_trn.labels, data_trn.N, d, best_features, model_LDA);
	(*all_errors).lsvm_bolster_error = lsvmBolster(data_trn.data, data_trn.labels, data_trn.N, d, best_features, N_bolster, seed);
	(*all_errors).ksvm_bolster_error = ksvmBolster(data_trn.data, data_trn.labels, data_trn.N, d, best_features, N_bolster, seed);

	(*all_errors).lda_loo_error = ldaLOO(data_trn.data, data_trn.labels, data_trn.N, data_trn.D, d);
	(*all_errors).lsvm_loo_error = lsvmLOO(data_trn.data, data_trn.labels, data_trn.N, data_trn.D, d);
	(*all_errors).ksvm_loo_error = ksvmLOO(data_trn.data, data_trn.labels, data_trn.N, data_trn.D, d);

	(*all_errors).lda_cvkfold_error = ldaCVkFold(data_trn.data, data_trn.labels, data_trn.N, data_trn.D, d, K, R, seed);
	(*all_errors).lsvm_cvkfold_error = lsvmCVkFold(data_trn.data, data_trn.labels, data_trn.N, data_trn.D, d, K, R, seed);
	(*all_errors).ksvm_cvkfold_error = ksvmCVkFold(data_trn.data, data_trn.labels, data_trn.N, data_trn.D, d, K, R, seed);

	(*all_errors).lda_boot632_error = ldaBoot632(data_trn.data, data_trn.labels, data_trn.N, data_trn.D, d, B, (*all_errors).lda_resub_error, seed);
	(*all_errors).lsvm_boot632_error = lsvmBoot632(data_trn.data, data_trn.labels, data_trn.N, data_trn.D, d, B, (*all_errors).lsvm_resub_error, seed);
	(*all_errors).ksvm_boot632_error = ksvmBoot632(data_trn.data, data_trn.labels, data_trn.N, data_trn.D, d, B, (*all_errors).ksvm_resub_error, seed);

	delete model_LDA.a;
	delete best_features;

	svmDestroy(model_LSVM, subdata_LSVM, &subcl_LSVM);
	svmDestroy(model_KSVM, subdata_KSVM, &subcl_KSVM);

	delete_2D_matrix(N_trn, D, data_trn.data);
	delete data_trn.labels;
	delete_2D_matrix(N_tst, D, data_tst.data);
	delete data_tst.labels;

	return;
}
