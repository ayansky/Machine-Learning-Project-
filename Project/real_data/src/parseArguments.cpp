/*
 * parseArguments.cpp
 *
 */

#include "parseArguments.h"

int  parse_common_arguments(
		char** dataFilename,
		char** outputFilename,
		int* N_trn,
		int* d,
		int* max_rep,
		int* round,
		int argc, char *argv[])
{
	if (argc <= 1)
	{
		printf("Too few parameters!\n");
	}

	//default value
	*dataFilename = NULL;
	*outputFilename = NULL;
	*N_trn = 30;
	*d = 1;
	*max_rep = 1;
	*round = 1;

	for (argc--, argv++; argc > 0; argc--, argv++)
		if (strcmp(*argv,"-i") == 0)
		{
			*(argv++) = ""; argc--;
			if (argc == 0)
			{
				printf("No input filename!\n");
				return(-1);
			}
			*dataFilename = *argv;
			*argv = "";
		}
		else if (strcmp(*argv,"-o") == 0)
		{
			*(argv++) = ""; argc--;
			if (argc == 0)
			{
				printf("No output filename!\n");
				return(-1);
			}
			*outputFilename = *argv;
			*argv = "";
		}
		else if (strcmp(*argv,"-tr") == 0)
		{
			*(argv++) = ""; argc--;
			if (argc == 0 || (sscanf(*argv,"%d", N_trn) <0))
			{
				printf("N_trn error!\n");
				return(-1);
			}
		}
		else if (strcmp(*argv,"-d") == 0)
		{
			*(argv++) = ""; argc--;
			if (argc == 0 || (sscanf(*argv,"%d", d) <0))
			{
				printf("d error!\n");
				return(-1);
			}
		}
		else if (strcmp(*argv,"-mr") == 0)
		{
			*(argv++) = ""; argc--;
			if (argc == 0 || (sscanf(*argv,"%d", max_rep) <0))
			{
				printf("max_rep error!\n");
				return(-1);
			}
		}
		else if (strcmp(*argv,"-r") == 0)
		{
			*(argv++) = ""; argc--;
			if (argc == 0 || (sscanf(*argv,"%d", round) <0))
			{
				printf("round error!\n");
				return(-1);
			}
		}
		else
		{
			printf("Unknown arguments!\n", *argv);
			return(-2);
		}

	if (*dataFilename == NULL)
	{
		printf("Must provide a filename!\n");
		return(-1);
	}
	if (*outputFilename == NULL)
	{
		printf("Must provide a filename!\n");
		return(-1);
	}

	return(0);
}
