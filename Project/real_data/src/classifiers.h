/*
 * classifiers.h
 *
 */

#ifndef CLASSIFIERS_H_
#define CLASSIFIERS_H_

#include "standardHeaders.h"
#include "matrixOperations.h"

struct LDA {
	double *a;
	double b;
};

void ldaTrn(double** X, int* y, int N, int d, int* ind, LDA* lda);
double ldaTst(double** X, int* y, int N, int d, int* ind, LDA lda);

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct svm_model *svmTrn(double** X, int* y, int N, int d, int* ind, int kernel_type, struct svm_node** subdata, struct svm_problem* subcl);
double svmTst(double** X, int* y, int N, int d, int* ind, svm_model* model);
void svmDestroy(struct svm_model*, struct svm_node*, struct svm_problem*);

#ifdef __cplusplus
extern "C" {
#endif

	struct svm_node
	{
		int index;
		double value;
		double quality;
	};

	struct svm_problem
	{
		int l;
		double *y;
		struct svm_node **x;
	};

	enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
	enum { LINEAR, POLY, RBF, SIGMOID };	/* kernel_type */

	struct svm_parameter
	{
		int svm_type;
		int kernel_type;
		double degree;	/* for poly */
		double gamma;	/* for poly/rbf/sigmoid */
		double coef0;	/* for poly/sigmoid */

		/* these are for training only */
		double cache_size; /* in MB */
		double eps;	/* stopping criteria */
		double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
		int nr_weight;		/* for C_SVC */
		int *weight_label;	/* for C_SVC */
		double* weight;		/* for C_SVC */
		double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
		double p;	/* for EPSILON_SVR */
		int shrinking;	/* use the shrinking heuristics */
	};

	struct svm_model
	{
		svm_parameter param;	// parameter
		int nr_class;		// number of classes, = 2 in regression/one class svm
		int l;			// total #SV
		svm_node **SV;		// SVs (SV[l])
		double **sv_coef;	// coefficients for SVs in decision functions (sv_coef[n-1][l])
		double *rho;		// constants in decision functions (rho[n*(n-1)/2])

		// for classification only

		int *label;		// label of each class (label[n])
		int *nSV;		// number of SVs for each class (nSV[n])
		// nSV[0] + nSV[1] + ... + nSV[n-1] = l
		// XXX
		int free_sv;		// 1 if svm_model is created by svm_load_model
		// 0 if svm_model is created by svm_train
	};

	struct svm_model *svm_train(const struct svm_problem *prob,
			const struct svm_parameter *param);

	int svm_save_model(const char*, const struct svm_model*);

	int get_sv(const struct svm_model*);

	struct svm_model *svm_load_model(const char*);

	double svm_predict(const struct svm_model*, const struct svm_node*);

	void svm_destroy_model(struct svm_model*);

	const char *svm_check_parameter(const struct svm_problem*, const struct svm_parameter*);

	void info(char*,...);

	void info_flush();

#ifdef __cplusplus
}
#endif


#endif /* CLASSIFIERS_H_ */
