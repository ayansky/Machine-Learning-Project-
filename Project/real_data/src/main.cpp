/*
 * main.cpp
 *
 */

#include "main.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

int main (int argc, char *argv[])
{
	//////////////////////// MPI ///////////////////////////////////
	int MYRANK=0, NP=1;   //DEFAULT assume total one node, current is node 0

#ifdef USE_MPI
	MPI_Status status;  // return status for receive
#endif

#ifdef USE_MPI
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &NP);
	MPI_Comm_rank(MPI_COMM_WORLD, &MYRANK);
#endif

	double dinit = 0.00;
	int i, j, flag, rep;

	int N_0; // samples in class 0
	int N_1; // samples in class 1
	int N; // total number of sample points
	int N_trn;
	int N_tst;
	int D; // total # of features
	int d = 1; // after feature selection
	int max_rep = 1;
	int round = 1;
	long seed = 0L;
	double temp;

	double** X;
	int* y;

	AllErrors all_errors;

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

	char *dataFilename;
	char *outputFilename;
	char outputFilenameFinal[80];

	FILE *fpIn;

	FILE* fp_lda_true_error;
	FILE* fp_lsvm_true_error;
	FILE* fp_ksvm_true_error;

	FILE* fp_lda_resub_error;
	FILE* fp_lsvm_resub_error;
	FILE* fp_ksvm_resub_error;

	FILE* fp_lda_bolster_error;
	FILE* fp_lsvm_bolster_error;
	FILE* fp_ksvm_bolster_error;

	FILE* fp_lda_loo_error;
	FILE* fp_lsvm_loo_error;
	FILE* fp_ksvm_loo_error;

	FILE* fp_lda_cvkfold_error;
	FILE* fp_lsvm_cvkfold_error;
	FILE* fp_ksvm_cvkfold_error;

	FILE* fp_lda_boot632_error;
	FILE* fp_lsvm_boot632_error;
	FILE* fp_ksvm_boot632_error;

	//===========================================//
	// Arguments are passed through command-line //
	//===========================================//
	switch (parse_common_arguments(
			&dataFilename,
			&outputFilename,
			&N_trn,
			&d,
			&max_rep,
			&round,
			argc, argv))
	{
	case -1:
	printf("Error in function parse_common_arguments()!\n");
	return(-1);
	case -2:
	printf("Usage printed\n"); // should add some usage printing function here
	return(0);
	default:
		printf("\n");
	}

	fpIn = fopen(dataFilename, "r");
	if(fpIn==NULL)
	{
		printf("cannot open data file in node %d!\n", MYRANK);
		return(-1);
	}

	//obtain the size information of two classes
	fscanf(fpIn,"%d, %d\r\n", &N_0, &N_1);

	// total number of features
	fscanf(fpIn,"%d\r\n", &D);

	//size_class1 = size_class0;
	N = N_0 + N_1;
	N_tst = N - N_trn;

	// initializing matrix of original data set
	X = make_2D_matrix(N, D, dinit);
	y = new int [N];

	// obtain the gene data from data file (rows: fetaures, columns: patients)
	// but we transpose it to: (rows: patients, columns: features)
	for (i=0; i<N; i++)
	{
		if (i<N_0)
			y[i] = 0;
		else
			y[i] = 1;
	}

	for (i=0; i<D; i++)
	{
		for (j=0; j<N; j++)
		{
			fscanf(fpIn, "%lf\t", &temp);
			X[j][i] = temp;
		}
		fscanf(fpIn, "\r\n");
	}

	fclose(fpIn);

	srand((unsigned)time(NULL)+MYRANK);
	seed = -abs((long)floor(rand()*time(NULL)+MYRANK+1));

	if (MYRANK==0)
	{
		sprintf(outputFilenameFinal, "%s_lda_true_error_round=%d.txt", outputFilename, round);
		fp_lda_true_error = fopen(outputFilenameFinal, "w");
		if(fp_lda_true_error==NULL)
		{
			printf("cannot open %s_lda_true_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_lsvm_true_error_round=%d.txt", outputFilename, round);
		fp_lsvm_true_error = fopen(outputFilenameFinal, "w");
		if(fp_lsvm_true_error==NULL)
		{
			printf("cannot open %s_lsvm_true_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_ksvm_true_error_round=%d.txt", outputFilename, round);
		fp_ksvm_true_error = fopen(outputFilenameFinal, "w");
		if(fp_ksvm_true_error==NULL)
		{
			printf("cannot open %s_ksvm_true_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_lda_resub_error_round=%d.txt", outputFilename, round);
		fp_lda_resub_error = fopen(outputFilenameFinal, "w");
		if(fp_lda_resub_error==NULL)
		{
			printf("cannot open %s_lda_resub_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_lsvm_resub_error_round=%d.txt", outputFilename, round);
		fp_lsvm_resub_error = fopen(outputFilenameFinal, "w");
		if(fp_lsvm_resub_error==NULL)
		{
			printf("cannot open %s_lsvm_resub_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_ksvm_resub_error_round=%d.txt", outputFilename, round);
		fp_ksvm_resub_error = fopen(outputFilenameFinal, "w");
		if(fp_ksvm_resub_error==NULL)
		{
			printf("cannot open %s_ksvm_resub_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_lda_bolster_error_round=%d.txt", outputFilename, round);
		fp_lda_bolster_error = fopen(outputFilenameFinal, "w");
		if(fp_lda_bolster_error==NULL)
		{
			printf("cannot open %s_lda_bolster_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_lsvm_bolster_error_round=%d.txt", outputFilename, round);
		fp_lsvm_bolster_error = fopen(outputFilenameFinal, "w");
		if(fp_lsvm_bolster_error==NULL)
		{
			printf("cannot open %s_lsvm_bolster_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_ksvm_bolster_error_round=%d.txt", outputFilename, round);
		fp_ksvm_bolster_error = fopen(outputFilenameFinal, "w");
		if(fp_ksvm_bolster_error==NULL)
		{
			printf("cannot open %s_ksvm_bolster_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_lda_loo_error_round=%d.txt", outputFilename, round);
		fp_lda_loo_error = fopen(outputFilenameFinal, "w");
		if(fp_lda_loo_error==NULL)
		{
			printf("cannot open %s_lda_loo_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_lsvm_loo_error_round=%d.txt", outputFilename, round);
		fp_lsvm_loo_error = fopen(outputFilenameFinal, "w");
		if(fp_lsvm_loo_error==NULL)
		{
			printf("cannot open %s_lsvm_loo_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_ksvm_loo_error_round=%d.txt", outputFilename, round);
		fp_ksvm_loo_error = fopen(outputFilenameFinal, "w");
		if(fp_ksvm_loo_error==NULL)
		{
			printf("cannot open %s_ksvm_loo_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_lda_cvkfold_error_round=%d.txt", outputFilename, round);
		fp_lda_cvkfold_error = fopen(outputFilenameFinal, "w");
		if(fp_lda_cvkfold_error==NULL)
		{
			printf("cannot open %s_lda_cvkfold_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_lsvm_cvkfold_error_round=%d.txt", outputFilename, round);
		fp_lsvm_cvkfold_error = fopen(outputFilenameFinal, "w");
		if(fp_lsvm_cvkfold_error==NULL)
		{
			printf("cannot open %s_lsvm_cvkfold_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_ksvm_cvkfold_error_round=%d.txt", outputFilename, round);
		fp_ksvm_cvkfold_error = fopen(outputFilenameFinal, "w");
		if(fp_ksvm_cvkfold_error==NULL)
		{
			printf("cannot open %s_ksvm_cvkfold_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_lda_boot632_error_round=%d.txt", outputFilename, round);
		fp_lda_boot632_error = fopen(outputFilenameFinal, "w");
		if(fp_lda_boot632_error==NULL)
		{
			printf("cannot open %s_lda_boot632_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_lsvm_boot632_error_round=%d.txt", outputFilename, round);
		fp_lsvm_boot632_error = fopen(outputFilenameFinal, "w");
		if(fp_lsvm_boot632_error==NULL)
		{
			printf("cannot open %s_lsvm_boot632_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_ksvm_boot632_error_round=%d.txt", outputFilename, round);
		fp_ksvm_boot632_error = fopen(outputFilenameFinal, "w");
		if(fp_ksvm_boot632_error==NULL)
		{
			printf("cannot open %s_ksvm_boot632_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
	}

	for (rep=MYRANK; rep<max_rep; rep+=NP)
	{
		//printf("rep=%d\n", rep);

		calculateErrors(X,
				y,
				N_trn,
				N_tst,
				D,
				d,
				&seed,
				&all_errors);

		lda_true_error = all_errors.lda_true_error;
		lsvm_true_error = all_errors.lsvm_true_error;
		ksvm_true_error = all_errors.ksvm_true_error;

		lda_resub_error = all_errors.lda_resub_error;
		lsvm_resub_error = all_errors.lsvm_resub_error;
		ksvm_resub_error = all_errors.ksvm_resub_error;

		lda_bolster_error = all_errors.lda_bolster_error;
		lsvm_bolster_error = all_errors.lsvm_bolster_error;
		ksvm_bolster_error = all_errors.ksvm_bolster_error;

		lda_loo_error = all_errors.lda_loo_error;
		lsvm_loo_error = all_errors.lsvm_loo_error;
		ksvm_loo_error = all_errors.ksvm_loo_error;

		lda_cvkfold_error = all_errors.lda_cvkfold_error;
		lsvm_cvkfold_error = all_errors.lsvm_cvkfold_error;
		ksvm_cvkfold_error = all_errors.ksvm_cvkfold_error;

		lda_boot632_error = all_errors.lda_boot632_error;
		lsvm_boot632_error = all_errors.lsvm_boot632_error;
		ksvm_boot632_error = all_errors.ksvm_boot632_error;

		if (MYRANK==0)
		{
			fprintf(fp_lda_true_error, "%f\n", lda_true_error);
			fprintf(fp_lsvm_true_error, "%f\n", lsvm_true_error);
			fprintf(fp_ksvm_true_error, "%f\n", ksvm_true_error);

			fprintf(fp_lda_resub_error, "%f\n", lda_resub_error);
			fprintf(fp_lsvm_resub_error, "%f\n", lsvm_resub_error);
			fprintf(fp_ksvm_resub_error, "%f\n", ksvm_resub_error);

			fprintf(fp_lda_bolster_error, "%f\n", lda_bolster_error);
			fprintf(fp_lsvm_bolster_error, "%f\n", lsvm_bolster_error);
			fprintf(fp_ksvm_bolster_error, "%f\n", ksvm_bolster_error);

			fprintf(fp_lda_loo_error, "%f\n", lda_loo_error);
			fprintf(fp_lsvm_loo_error, "%f\n", lsvm_loo_error);
			fprintf(fp_ksvm_loo_error, "%f\n", ksvm_loo_error);

			fprintf(fp_lda_cvkfold_error, "%f\n", lda_cvkfold_error);
			fprintf(fp_lsvm_cvkfold_error, "%f\n", lsvm_cvkfold_error);
			fprintf(fp_ksvm_cvkfold_error, "%f\n", ksvm_cvkfold_error);

			fprintf(fp_lda_boot632_error, "%f\n", lda_boot632_error);
			fprintf(fp_lsvm_boot632_error, "%f\n", lsvm_boot632_error);
			fprintf(fp_ksvm_boot632_error, "%f\n", ksvm_boot632_error);

#ifdef USE_MPI
			flag = ((max_rep-rep)>(max_rep)%NP ? NP : max_rep%NP);
			for (j=1; j<flag; j++)
			 /*MPI implementation could be here */
#endif
		}
#ifdef USE_MPI
		else
            /*MPI implementation could be here */
#endif
	}

	if (MYRANK==0)
	{
		fclose(fp_lda_true_error);
		fclose(fp_lsvm_true_error);
		fclose(fp_ksvm_true_error);
		fclose(fp_lda_resub_error);
		fclose(fp_lsvm_resub_error);
		fclose(fp_ksvm_resub_error);
		fclose(fp_lda_bolster_error);
		fclose(fp_lsvm_bolster_error);
		fclose(fp_ksvm_bolster_error);
		fclose(fp_lda_loo_error);
		fclose(fp_lsvm_loo_error);
		fclose(fp_ksvm_loo_error);
		fclose(fp_lda_cvkfold_error);
		fclose(fp_lsvm_cvkfold_error);
		fclose(fp_ksvm_cvkfold_error);
		fclose(fp_lda_boot632_error);
		fclose(fp_lsvm_boot632_error);
		fclose(fp_ksvm_boot632_error);
	}

	delete_2D_matrix(N, D, X);
	delete y;

#ifdef USE_MPI
	/*MPI implementation could be here */
#endif

	return 0;
}

