/*
 * print.cpp
 *
 */

#include "print.h"

void PrintDouble(double** data, int rows, int cols, char* name)
{
	int i, j;
	FILE *fp;

	fp = fopen(name, "w");
	if(fp==NULL)
	{
		printf("cannot open output file\n");
		return;
	}

	for (i=0; i<rows; i++)
	{
		for (j=0; j<cols; j++)
		{
			fprintf(fp, "%f\t", data[i][j]);
		}
		fprintf(fp, "\n");
	}

	return;
}
