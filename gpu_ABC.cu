/* ABC algorithm coded using C programming language */

/* Artificial Bee Colony (ABC) is one of the most recently defined algorithms by Dervis Karaboga in 2005,
motivated by the intelligent behavior of honey bees. */

/* Referance Papers*/

/*D. Karaboga, AN IDEA BASED ON HONEY BEE SWARM FOR NUMERICAL OPTIMIZATION,TECHNICAL REPORT-TR06, Erciyes University, Engineering Faculty, Computer Engineering Department 2005.*/

/*D. Karaboga, B. Basturk, A powerful and Efficient Algorithm for Numerical Function Optimization: Artificial Bee Colony (ABC) Algorithm, Journal of Global Optimization, Volume:39, Issue:3,pp:459-171, November 2007,ISSN:0925-5001 , doi: 10.1007/s10898-007-9149-x */

/*D. Karaboga, B. Basturk, On The Performance Of Artificial Bee Colony (ABC) Algorithm, Applied Soft Computing,Volume 8, Issue 1, January 2008, Pages 687-697. */

/*D. Karaboga, B. Akay, A Comparative Study of Artificial Bee Colony Algorithm,  Applied Mathematics and Computation, 214, 108-132, 2009. */

/*Copyright © 2009 Erciyes University, Intelligent Systems Research Group, The Dept. of Computer Engineering*/

/*Contact:
Dervis Karaboga (karaboga@erciyes.edu.tr )
Bahriye Basturk Akay (bahriye@erciyes.edu.tr)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas.h>
// #include <conio.h> Only works on windows
//#include <ncurses.h> // Alternative to conio.h for Unix Systems
#include <time.h>
#include <curand_kernel.h>

/* Control Parameters of ABC algorithm*/
#define NP 64             /* The number of colony size (employed bees+onlooker bees)*/ // Default = 40
#define FoodNumber NP / 2 /*The number of food sources equals the half of the colony size*/
#define limit 100         /*A food source which could not be improved through "limit" trials is abandoned by its employed bee*/
#define maxCycle 3000     /*The number of cycles for foraging {a stopping criteria}*/

/* Problem specific variables*/
#define D 50     /*The number of parameters of the problem to be optimized*/
#define lb -5.12 /*lower bound of the parameters. */
#define ub 5.12  /*upper bound of the parameters. lb and ub can be defined as arrays for the problems of which parameters have different bounds*/

#define runtime 30 /*Algorithm can be run many times in order to see its robustness*/
#define colony_size 2

double Foods[FoodNumber][D]; /*Foods is the population of food sources. Each row of Foods matrix is a vector holding D parameters to be optimized. The number of rows of Foods matrix equals to the FoodNumber*/
double f[FoodNumber];        /*f is a vector holding objective function values associated with food sources */
double fitness[FoodNumber];  /*fitness is a vector holding fitness (quality) values associated with food sources*/
double trial[FoodNumber];    /*trial is a vector holding trial numbers through which solutions can not be improved*/
double prob[FoodNumber];     /*prob is a vector holding probabilities of food sources (solutions) to be chosen*/
double solution[D];          /*New solution (neighbour) produced by v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) j is a randomly chosen parameter and k is a randomlu chosen solution different from i*/
double ObjValSol;            /*Objective function value of new solution*/
double FitnessSol;           /*Fitness value of new solution*/
//int neighbour, param2change; /*param2change corrresponds to j, neighbour corresponds to k in equation v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})*/
double GlobalMin[1];            /*Optimum solution obtained by ABC algorithm*/
double GlobalParams[D];      /*Parameters of the optimum solution*/
double GlobalMins[runtime];  /*GlobalMins holds the GlobalMin of each run in multiple runs*/
//double r;                    /*a random number in the range [0,1)*/


double solution_array[FoodNumber*D];
/*a function pointer returning double and taking a D-dimensional array as argument */
/*If your function takes additional arguments then change function pointer definition and lines calling "...=function(solution);" in the code*/
typedef double (*FunctionCallback)(double sol[D]);

/*benchmark functions */
__device__ __host__ double sphere(double sol[D]);
__device__ __host__ double Rosenbrock(double sol[D]);
__device__ __host__ double Griewank(double sol[D]);
__device__ __host__ double Rastrigin(double sol[D]);

/*Write your own objective function name instead of sphere*/
__device__ FunctionCallback function = &Rastrigin;
 FunctionCallback function_host = &Rastrigin;
 
/*Fitness function*/
__device__ __host__ double CalculateFitness(double fun)
{
    double result = 0;
    if (fun >= 0)
    {
        result = 1 / (fun + 1);
    }
    else
    {
        result = 1 + fabs(fun);
    }
    return result;
}

/*The best food source is memorized*/
void MemorizeBestSource()
{
    int i, j;

    for (i = 0; i < FoodNumber; i++)
    {
        if (f[i] < GlobalMin[0])
        {
            GlobalMin[0] = f[i];
            for (j = 0; j < D; j++)
                GlobalParams[j] = Foods[i][j];
        }
    }
}


__device__ void MemorizeBestSource_gpu(double* gpu_f, double* gpu_GlobalMin, double* gpu_GlobalParams, double* gpu_solution_array, int my_y)
{
    int j;

    
    if (gpu_f[my_y] < gpu_GlobalMin[0])
    {
        gpu_GlobalMin[0] = gpu_f[my_y];
        for (j = 0; j < D; j++)
            gpu_GlobalParams[j] = gpu_solution_array[my_y*D + j];
    }
}
/*Variables are initialized in the range [lb,ub]. If each parameter has different range, use arrays lb[j], ub[j] instead of lb and ub */
/* Counters of food sources are also initialized in this function*/
void init(int index)
{
    int j;
	double r;
    for (j = 0; j < D; j++)
    {
        r = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
        Foods[index][j] = r * (ub - lb) + lb;
		solution_array[index*D + j] = r * (ub - lb) + lb;
		
        solution[j] = Foods[index][j];
    }
    f[index] = function_host(solution);
    fitness[index] = CalculateFitness(f[index]);
    trial[index] = 0;
}


__device__ void init_gpu(curandState *state,int* index, double* gpu_solution_array, double* solution,double* gpu_f,double* gpu_fitness,double* gpu_trial, int my_y)
{
    int j;
	double randomValue;
    for (j = 0; j < 50; j++)
    {
        //r = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
		randomValue = curand_uniform(&state[index[0]]);
        //Foods[index[0]][j] = r * (ub - lb) + lb;
		gpu_solution_array [index[0]*D + j] = randomValue * (ub - lb) + lb;;
		
        solution[j] = gpu_solution_array [index[0]*50 + j];
    }
    gpu_f[index[0]] = function(solution);
    gpu_fitness[index[0]] = CalculateFitness(gpu_f[index[0]]);
    gpu_trial[my_y] = 0;
}



/*All food sources are initialized */
void initial()
{
    int i;
    for (i = 0; i < FoodNumber; i++)
    {
        init(i);
    }
    GlobalMin[0] = f[0];
    for (i = 0; i < D; i++)
        GlobalParams[i] = Foods[0][i];
}

__device__ void SendEmployedBees(curandState *state,double* gpu_solution_array, int my_y, double* solution, double ObjValSol, double FitnessSol, double* gpu_fitness, double* gpu_trial, double* gpu_f, int* gpu_maxtrial, int* gpu_maxtrialindex)
{
    int j;
    /*Employed Bee Phase*/

	int neighbour, param2change;
	double randomValue;	
    
        /*The parameter to be changed is determined randomly*/
        //r = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
		randomValue = curand_uniform(&state[my_y]);
        param2change = (int)(randomValue * D);

        /*A randomly chosen solution is used in producing a mutant solution of the solution i*/
        //r = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
		randomValue = curand_uniform(&state[my_y]);
        neighbour = (int)(randomValue * FoodNumber);

        /*Randomly selected solution must be different from the solution i*/
        while (neighbour == my_y)
        {
            //r = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
			randomValue = curand_uniform(&state[my_y]);
            neighbour = (int)(randomValue * FoodNumber);
        }
        
		//double solution[D]; 
		//double ObjValSol;
		//double FitnessSol;
		
		for (j = 0; j < D; j++)
			//solution[j] = Foods[i][j];
			solution[j] = gpu_solution_array[my_y*50+j];
        /*v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) */
        //r = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
		randomValue = curand_uniform(&state[my_y]);
        //solution[param2change] = Foods[i][param2change] + (Foods[i][param2change] - Foods[neighbour][param2change]) * (r - 0.5) * 2;
		solution[param2change] = gpu_solution_array[my_y*50+param2change] + (gpu_solution_array[my_y*50+param2change] - gpu_solution_array[neighbour*50+param2change]) * (randomValue - 0.5) * 2;
        /*if generated parameter value is out of boundaries, it is shifted onto the boundaries*/
        if (solution[param2change] < lb)
            solution[param2change] = lb;
        if (solution[param2change] > ub)
            solution[param2change] = ub;
		
		
		
        ObjValSol = function(solution);
        FitnessSol = CalculateFitness(ObjValSol);

        /*a greedy selection is applied between the current solution i and its mutant*/
        if (FitnessSol > gpu_fitness[my_y])
        {
            /*If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i*/
            gpu_trial[my_y] = 0;
            for (j = 0; j < D; j++)
                //Foods[i][j] = solution[j];
				gpu_solution_array[my_y*50+j] = solution[j];
            gpu_f[my_y] = ObjValSol;
            gpu_fitness[my_y] = FitnessSol;
        }
        else
        { /*if the solution i can not be improved, increase its trial counter*/
            gpu_trial[my_y] = gpu_trial[my_y] + 1;
        }

    /*end of employed bee phase*/
}

/* A food source is chosen with the probability which is proportioal to its quality*/
/*Different schemes can be used to calculate the probability values*/
/*For example prob(i)=fitness(i)/sum(fitness)*/
/*or in a way used in the metot below prob(i)=a*fitness(i)/max(fitness)+b*/
/*probability values are calculated by using fitness values and normalized by dividing maximum fitness value*/
__device__ void CalculateProbabilities(double* gpu_fitness, int my_y, double* gpu_prob, int maxfit)
{
    //int i;
    //double maxfit;
    //maxfit = gpu_fitness[0];
    
    if (gpu_fitness[my_y] > maxfit)
          maxfit = gpu_fitness[my_y];
    

    
    gpu_prob[my_y] = (0.9 * (gpu_fitness[my_y] / maxfit)) + 0.1;
    
}

__device__ void SendOnlookerBees(curandState *state,double* gpu_prob, double* gpu_solution_array,int my_y, double* solution, double ObjValSol, double FitnessSol, double* gpu_fitness, double* gpu_trial, double* gpu_f, int* gpu_maxtrial, int* gpu_maxtrialindex)
{

    int j;
	double randomValue;
	
	int neighbour, param2change;
    //i = 0;
    //t = 0;
    /*onlooker Bee Phase*/
    //while (t < FoodNumber)
    //{

        //r = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
		randomValue = curand_uniform(&state[my_y]);
        if (randomValue < gpu_prob[my_y]) /*choose a food source depending on its probability to be chosen*/
        {
            //++;

            /*The parameter to be changed is determined randomly*/
            //r = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
			randomValue = curand_uniform(&state[my_y]);
            param2change = (int)(randomValue * D);

            /*A randomly chosen solution is used in producing a mutant solution of the solution i*/
            //r = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
			randomValue = curand_uniform(&state[my_y]);
            neighbour = (int)(randomValue * FoodNumber);

            /*Randomly selected solution must be different from the solution i*/
            while (neighbour == my_y)
            {
                //r = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
				randomValue = curand_uniform(&state[my_y]);
                neighbour = (int)(randomValue * FoodNumber);
            }
            for (j = 0; j < D; j++)
                //solution[j] = Foods[i][j];
				solution[j] = gpu_solution_array[my_y*50+j];
            /*v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) */
            //r = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
			randomValue = curand_uniform(&state[my_y]);
            //solution[param2change] = Foods[i][param2change] + (Foods[i][param2change] - Foods[neighbour][param2change]) * (r - 0.5) * 2;
			solution[param2change] = gpu_solution_array[my_y*50+param2change] + (gpu_solution_array[my_y*50+param2change] - gpu_solution_array[neighbour*50+param2change]) * (randomValue - 0.5) * 2;
            /*if generated parameter value is out of boundaries, it is shifted onto the boundaries*/
            if (solution[param2change] < lb)
                solution[param2change] = lb;
            if (solution[param2change] > ub)
                solution[param2change] = ub;
            ObjValSol = function(solution);
            FitnessSol = CalculateFitness(ObjValSol);

            /*a greedy selection is applied between the current solution i and its mutant*/
            if (FitnessSol > gpu_fitness[my_y])
            {
                /*If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i*/
                gpu_trial[my_y] = 0;
                for (j = 0; j < D; j++)
                    //Foods[i][j] = solution[j];
					gpu_solution_array[my_y*50+j] = solution[j];
                gpu_f[my_y] = ObjValSol;
                gpu_fitness[my_y] = FitnessSol;
            }
            else
            { /*if the solution i can not be improved, increase its trial counter*/
                gpu_trial[my_y] = gpu_trial[my_y] + 1;
            }
        } /*if */
        //i++;
        //if (i == FoodNumber)
            //i = 0;
    //} /*while*/

    /*end of onlooker bee phase     */
}

/*determine the food sources whose trial counter exceeds the "limit" value. In Basic ABC, only one scout is allowed to occur in each cycle*/
__device__ void SendScoutBees(curandState *state,int my_y, int* gpu_maxtrial,int* gpu_maxtrialindex, double* gpu_trial, double* gpu_solution_array, double* solution, double* gpu_f, double* gpu_fitness)
{
    //int maxtrialindex, maxtrial, i;
    //maxtrialindex = 0;
	//maxtrial = 0;
	
    if (gpu_trial[my_y] > gpu_maxtrial[0])
          gpu_maxtrialindex[0] = my_y;
    
    if (gpu_maxtrial[0] >= limit)
    {
        init_gpu(state, gpu_maxtrialindex, gpu_solution_array, solution, gpu_f, gpu_fitness, gpu_trial, my_y);
    }
}



__global__ void find_optimized_solution(curandState *state, int seed,double *gpu_solution_array, double *gpu_f, double *gpu_fitness, double* gpu_GlobalMin, double* gpu_GlobalParams,int* gpu_maxtrial, int* gpu_maxtrialindex, double*  gpu_prob, double* gpu_trial){
	int my_y, iter;
	//my_x = blockIdx.x*blockDim.x + threadIdx.x;
	my_y = blockIdx.y*blockDim.y + threadIdx.y;
	
	curand_init(seed, my_y, 0, &state[my_y]);
	
	double maxfit = gpu_fitness[0];
	
	double solution[D]; 
	double ObjValSol;
	double FitnessSol;
	
	//double prob[FoodNumber];
	
	for (iter = 0; iter < maxCycle; iter++)
        {
            SendEmployedBees(state,gpu_solution_array, my_y,solution, ObjValSol, FitnessSol, gpu_fitness, gpu_trial, gpu_f, gpu_maxtrial, gpu_maxtrialindex);
            CalculateProbabilities(gpu_fitness, my_y,  gpu_prob,  maxfit);
            SendOnlookerBees(state, gpu_prob, gpu_solution_array, my_y, solution, ObjValSol,  FitnessSol,  gpu_fitness,  gpu_trial,  gpu_f,  gpu_maxtrial,  gpu_maxtrialindex);
            MemorizeBestSource_gpu(gpu_f,  gpu_GlobalMin,  gpu_GlobalParams, gpu_solution_array,  my_y);
            SendScoutBees(state, my_y, gpu_maxtrial, gpu_maxtrialindex, gpu_trial, gpu_solution_array, solution,  gpu_f,  gpu_fitness);
        }
	 
}






/*Main program of the ABC algorithm*/
int main()
{

    printf("Running ABC serially on CPU with Colony size of %d\n", NP);
    
    int run;
    double mean;
    mean = 0;
    srand(time(NULL));

    struct timespec start, stop; 
	double t;
	
	
	
	double *solution_array = (double*)malloc(sizeof(int)*FoodNumber*D);
	//double *f = (double*)malloc(sizeof(int)*FoodNumber*D);
	//double *fitness = (double*)malloc(sizeof(int)*FoodNumber*D);
	
	double *optimized_solution = (double*)malloc(sizeof(int)*1);
	
	
	
	
	double *gpu_solution_array;
	double *gpu_f;
	double *gpu_fitness;
	
	double *gpu_GlobalMin;
	double *gpu_GlobalParams;

	int *gpu_maxtrial;
	int *gpu_maxtrialindex;
	
	double *gpu_prob;
	
	double *gpu_trial;
	
	cudaMalloc((void**)&gpu_solution_array, sizeof(int)*FoodNumber*D);
	cudaMalloc((void**)&gpu_f, sizeof(int)*FoodNumber*D);
	cudaMalloc((void**)&gpu_fitness, sizeof(int)*FoodNumber*D);
	
	cudaMalloc((void**)&gpu_maxtrial, sizeof(int)*1);
	cudaMalloc((void**)&gpu_maxtrialindex, sizeof(int)*1);
	
	cudaMalloc((void**)&gpu_GlobalMin, sizeof(int)*1);
	cudaMalloc((void**)&gpu_GlobalParams, sizeof(int)*D);
	cudaMalloc((void**)&gpu_prob, sizeof(int)*D);
	
	cudaMalloc((void**)&gpu_trial, sizeof(int)*FoodNumber);
	
	
	
	dim3 dimGrid(2);
	dim3 dimBlock(32);
	
	
	curandState *devStates;
    cudaMalloc((void **)&devStates, NP * sizeof(curandState));
	int seed = 1234;
	
	
    // measure the start time here
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

    for (run = 0; run < runtime; run++)
    {

        initial();
        MemorizeBestSource();
		
		
		cudaMemcpy(gpu_solution_array, solution_array, sizeof(int)*FoodNumber*D, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_f, f, sizeof(int)*FoodNumber, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_fitness, fitness, sizeof(int)*FoodNumber, cudaMemcpyHostToDevice);
		
		
		cudaMemcpy(gpu_GlobalMin, GlobalMin, sizeof(int)*1, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_GlobalParams, GlobalParams, sizeof(int)*D, cudaMemcpyHostToDevice);
		
		cudaMemcpy(gpu_trial, trial, sizeof(int)*FoodNumber, cudaMemcpyHostToDevice);
		find_optimized_solution<<<dimGrid, dimBlock>>>(devStates, seed, gpu_solution_array, gpu_f, gpu_fitness, gpu_GlobalMin, gpu_GlobalParams, gpu_maxtrial, gpu_maxtrialindex, gpu_prob, gpu_trial);
		
		cudaMemcpy(optimized_solution, gpu_GlobalMin, sizeof(int)*1, cudaMemcpyDeviceToHost);
		
        //for (iter = 0; iter < maxCycle; iter++)
        //{
        //    SendEmployedBees();
        //   CalculateProbabilities();
        //    SendOnlookerBees();
        //    MemorizeBestSource();
        //    SendScoutBees();
        //}
        // for (j = 0; j < D; j++)
        // {
        //     printf("GlobalParam[%d]: %f\n", j + 1, GlobalParams[j]);
        // }
        printf("%d. run: %e \n", run + 1, GlobalMin);
        GlobalMins[run] = optimized_solution[0];
        mean = mean + GlobalMin[0];
    }
    mean = mean / runtime;
    printf("Means of %d runs: %e\n", runtime, mean);

    // measure the end time here
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
    t = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;

    printf("Time taken to run %d iterations: %f sec\n", runtime, t);

    t /= runtime;
    
    // print out the execution time here
    printf("Average Time taken to run each iterations: %f sec\n", t);

    //getch();
	
	
	
	
	free(solution_array);
	free(optimized_solution);
	//free(c);
	cudaFree(gpu_GlobalMin);  
	cudaFree(gpu_GlobalParams);  
	cudaFree(gpu_maxtrial);
	cudaFree(gpu_maxtrialindex);
	cudaFree(gpu_fitness);
	cudaFree(gpu_f);
	cudaFree(gpu_solution_array);
	
	cudaFree(devStates);
	
	return 0;
	
	
}

double sphere(double sol[D])
{
    int j;
    double top = 0;
    for (j = 0; j < D; j++)
    {
        top = top + sol[j] * sol[j];
    }
    return top;
}

double Rosenbrock(double sol[D])
{
    int j;
    double top = 0;
    for (j = 0; j < D - 1; j++)
    {
        top = top + 100 * pow((sol[j + 1] - pow((sol[j]), (double)2)), (double)2) + pow((sol[j] - 1), (double)2);
    }
    return top;
}

double Griewank(double sol[D])
{
    int j;
    double top1, top2, top;
    top = 0;
    top1 = 0;
    top2 = 1;
    for (j = 0; j < D; j++)
    {
        top1 = top1 + pow((sol[j]), (double)2);
        top2 = top2 * cos((((sol[j]) / sqrt((double)(j + 1))) * M_PI) / 180);
    }
    top = (1 / (double)4000) * top1 - top2 + 1;
    return top;
}

double Rastrigin(double sol[D])
{
    int j;
    double top = 0;

    for (j = 0; j < D; j++)
    {
        top = top + (pow(sol[j], (double)2) - 10 * cos(2 * M_PI * sol[j]) + 10);
    }
    return top;
}