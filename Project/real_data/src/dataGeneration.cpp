/*
 * dataGeneration.cpp
 *
 */

#include "dataGeneration.h"

void dataGeneration(double** X,
		int* y,
		int N_trn,
		int N_tst,
		int D,
		long* seed,
		SimulationData* data_trn,
		SimulationData* data_tst)
{
	double dinit = 0.00;
	int i, j, k;
	int N_trn_0; // training samples in class 0
	int N_trn_1; // training samples in class 1
	int N = N_trn+N_tst;
	int N_0 = 0;
	int N_1 = 0;

	int* I0;
	int* I1;

	(*data_trn).N = N_trn;;
	(*data_trn).D = D;

	(*data_tst).N = N_tst;;
	(*data_tst).D = D;

	for (i=0; i<N; i++)
		if (y[i]==0)
			N_0++;
		else
			N_1++;

	N_trn_0 = (int)floor((float)(*data_trn).N*(float)N_0/(float)N + 0.5);
	N_trn_1 = (*data_trn).N - N_trn_0;



	I0 = new int [N_0];
	I1 = new int [N_1];

	for (i=0; i<N_0; i++)
		I0[i] = i;

	for (i=0; i<N_1; i++)
		I1[i] = i;

	shuffleArray(N_0, seed, I0);
	shuffleArray(N_1, seed, I1);

	for(i=0; i<(*data_trn).N; i++)
	{
		if (i<N_trn_0)
		{
			for(j=0; j<(*data_trn).D; j++)
				(*data_trn).data[i][j] = X[I0[i]][j];
			(*data_trn).labels[i] = y[I0[i]];
		}
		else
		{
			for(j=0; j<(*data_trn).D; j++)
				(*data_trn).data[i][j] = X[I1[i-N_trn_0]+N_0][j];
			(*data_trn).labels[i] = y[I1[i-N_trn_0]+N_0];
		}
	}

	for(i=0; i<(*data_tst).N; i++)
	{
		if (i<N_0-N_trn_0)
		{
			for(j=0; j<(*data_trn).D; j++)
				(*data_tst).data[i][j] = X[I0[i+N_trn_0]][j];
			(*data_tst).labels[i] = y[I0[i+N_trn_0]];
		}
		else
		{
			for(j=0; j<(*data_trn).D; j++)
				(*data_tst).data[i][j] = X[I1[i-(N_0-N_trn_0)+N_trn_1]+N_0][j];
			(*data_tst).labels[i] = y[I1[i-(N_0-N_trn_0)+N_trn_1]+N_0];
		}
	}


	delete I0;
	delete I1;

	return;
}
