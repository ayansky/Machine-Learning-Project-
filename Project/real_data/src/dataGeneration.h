/*
 * dataGeneration.h
 *
 */

#ifndef DATAGENERATION_H_
#define DATAGENERATION_H_

#include "standardHeaders.h"
#include "utilities.h"
#include "matrixOperations.h"
#include "random.h"

struct SimulationData {
  double** data;
  int* labels; /* labels for samples */
  int N;
  int D;
};

void dataGeneration(double** X,
		int* y,
		int N_trn,
		int N_tst,
		int D,
		long* seed,
		SimulationData* data_trn,
		SimulationData* data_tst);

#endif /* DATAGENERATION_H_ */
