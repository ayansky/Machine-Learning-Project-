/*
 * parseArguments.h
 *
 */

#ifndef PARSEARGUMENTS_H_
#define PARSEARGUMENTS_H_

#include "standardHeaders.h"

int  parse_common_arguments(
		char** dataFilename,
		char** outputFilename,
		int* N_trn,
		int* d,
		int* max_rep,
		int* round,
		int argc, char *argv[]);

#endif /* PARSEARGUMENTS_H_ */
