#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

/* Control Parameters of PABC */
#define P 2 // Number of Processors


/* Control Parameters of ABC algorithm*/
#define NP 80             /* The number of colony size (employed bees+onlooker bees)*/
#define FoodNumber NP / 2 /*The number of food sources equals the half of the colony size*/
#define SwarmFoodNumber FoodNumber / P // The number of food sources per swarm
#define limit 100         /*A food source which could not be improved through "limit" trials is abandoned by its employed bee*/
#define maxCycle 3000     /*The number of cycles for foraging {a stopping criteria}*/

/* Problem specific variables*/
#define D 50     /*The number of parameters of the problem to be optimized*/
#define lb -5.12 /*lower bound of the parameters. */
#define ub 5.12  /*upper bound of the parameters. lb and ub can be defined as arrays for the problems of which parameters have different bounds*/

#define runtime 30 /*Algorithm can be run many times in order to see its robustness*/

double Foods[FoodNumber][D]; /*Foods is the population of food sources. Each row of Foods matrix is a vector holding D parameters to be optimized. The number of rows of Foods matrix equals to the FoodNumber*/
double f[FoodNumber];        /*f is a vector holding objective function values associated with food sources */
double fitness[FoodNumber];  /*fitness is a vector holding fitness (quality) values associated with food sources*/
double trial[FoodNumber];    /*trial is a vector holding trial numbers through which solutions can not be improved*/
double prob[FoodNumber];     /*prob is a vector holding probabilities of food sources (solutions) to be chosen*/
double solution[P][D];          /*New solution (neighbour) produced by v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) j is a randomly chosen parameter and k is a randomlu chosen solution different from i*/
// double ObjValSol;            /*Objective function value of new solution*/
// double FitnessSol;           /*Fitness value of new solution*/
// int neighbour, param2change; /*param2change corrresponds to j, neighbour corresponds to k in equation v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})*/
// double GlobalMin;            /*Optimum solution obtained by ABC algorithm*/
double GlobalMin[P];            
// double GlobalParams[D];      /*Parameters of the optimum solution*/
double GlobalParams[P][D];
double GlobalMins[runtime];  /*GlobalMins holds the GlobalMin of each run in multiple runs*/
// double r;                    /*a random number in the range [0,1)*/

/*a function pointer returning double and taking a D-dimensional array as argument */
/*If your function takes additional arguments then change function pointer definition and lines calling "...=function(solution);" in the code*/
typedef double (*FunctionCallback)(double sol[D]);

/*benchmark functions */
double sphere(double sol[D]);
double Rosenbrock(double sol[D]);
double Griewank(double sol[D]);
double Rastrigin(double sol[D]);

/*Write your own objective function name instead of sphere*/
FunctionCallback function = &Rastrigin;

// typedef struct{
//     double Foods[FoodNumber][D]; /*Foods is the population of food sources. Each row of Foods matrix is a vector holding D parameters to be optimized. The number of rows of Foods matrix equals to the FoodNumber*/
//     double f[FoodNumber];        /*f is a vector holding objective function values associated with food sources */
//     double fitness[FoodNumber];  /*fitness is a vector holding fitness (quality) values associated with food sources*/
//     double trial[FoodNumber];    /*trial is a vector holding trial numbers through which solutions can not be improved*/
//     double prob[FoodNumber];     /*prob is a vector holding probabilities of food sources (solutions) to be chosen*/
// };

typedef struct {
    int threadIdx;
    int startIdx;
    int endIdx;
    unsigned int seed; // Seed for rand_r
} thread_data;

/*Fitness function*/
double CalculateFitness(double fun)
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
void MemorizeBestSourceForAll()
{
    int i, j, k;

    for  (i = 0; i < P; i++)
    {
        for (j = i * SwarmFoodNumber; j < (i + 1) * SwarmFoodNumber; j++)
        {
            if (f[j] < GlobalMin[i])
            {
                GlobalMin[i] = f[j];
                for (k = 0; k < D; k++)
                    GlobalParams[i][k] = Foods[j][k];
            }
        }
    }
}

void MemorizeBestSource(thread_data* thread_data)
{
    int i, j;

    for (i = thread_data->startIdx; i < thread_data->endIdx; i++)
    {
        if (f[i] < GlobalMin[thread_data->threadIdx])
        {
            GlobalMin[thread_data->threadIdx] = f[i];
            for (j = 0; j < D; j++)
                GlobalParams[thread_data->threadIdx][j] = Foods[i][j];
        }
    }
}

/*Variables are initialized in the range [lb,ub]. If each parameter has different range, use arrays lb[j], ub[j] instead of lb and ub */
/* Counters of food sources are also initialized in this function*/
void init(int index, unsigned int *seed, int threadIdx)
{
    int j;
    for (j = 0; j < D; j++)
    {
        double r = ((double)rand_r(seed) / ((double)(RAND_MAX) + (double)(1)));
        Foods[index][j] = r * (ub - lb) + lb;
        solution[threadIdx][j] = Foods[index][j];
    }
    f[index] = function(solution[threadIdx]);
    fitness[index] = CalculateFitness(f[index]);
    trial[index] = 0;
}


/*All food sources are initialized */
void initial(thread_data* thread_data)
{
    int i, j;
    for (i = 0; i < P; i++)
    {
        for (j = i * SwarmFoodNumber; j < (i + 1) * SwarmFoodNumber; j++)
        {
            init(j, &thread_data[i].seed, thread_data[i].threadIdx);
        }
    }
    for (i = 0; i < P; i++)
    {
        GlobalMin[i] = f[i * SwarmFoodNumber];
        for (j = 0; j < D; j++)
            GlobalParams[i][j] = Foods[i * SwarmFoodNumber][j];
        }
}


void SendEmployedBees(thread_data* thread_data)
{
    int i, j;
    unsigned int *seed = &thread_data->seed;
    /*Employed Bee Phase*/
    for (i = thread_data->startIdx; i < thread_data->endIdx; i++)
    {
        /*The parameter to be changed is determined randomly*/
        double r = ((double)rand_r(seed) / ((double)(RAND_MAX) + (double)(1)));
        int param2change = (int)(r * D);

        /*A randomly chosen solution is used in producing a mutant solution of the solution i*/
        r = ((double)rand_r(seed) / ((double)(RAND_MAX) + (double)(1)));
        int neighbour = thread_data->startIdx + (int)(r * SwarmFoodNumber);

        /*Randomly selected solution must be different from the solution i*/
        while (neighbour == i)
        {
            r = ((double)rand_r(seed) / ((double)(RAND_MAX) + (double)(1)));
            neighbour = thread_data->startIdx + (int)(r * SwarmFoodNumber);
        }
        for (j = 0; j < D; j++)
            solution[thread_data->threadIdx][j] = Foods[i][j];

        /*v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) */
        r = ((double)rand_r(seed) / ((double)(RAND_MAX) + (double)(1)));
        solution[thread_data->threadIdx][param2change] = Foods[i][param2change] + (Foods[i][param2change] - Foods[neighbour][param2change]) * (r - 0.5) * 2;

        /*if generated parameter value is out of boundaries, it is shifted onto the boundaries*/
        if (solution[thread_data->threadIdx][param2change] < lb)
            solution[thread_data->threadIdx][param2change] = lb;
        if (solution[thread_data->threadIdx][param2change] > ub)
            solution[thread_data->threadIdx][param2change] = ub;
        double ObjValSol = function(solution[thread_data->threadIdx]);
        double FitnessSol = CalculateFitness(ObjValSol);

        /*a greedy selection is applied between the current solution i and its mutant*/
        if (FitnessSol > fitness[i])
        {
            /*If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i*/
            trial[i] = 0;
            for (j = 0; j < D; j++)
                Foods[i][j] = solution[thread_data->threadIdx][j];
            f[i] = ObjValSol;
            fitness[i] = FitnessSol;
        }
        else
        { /*if the solution i can not be improved, increase its trial counter*/
            trial[i] = trial[i] + 1;
        }
    }

    /*end of employed bee phase*/
}

/* A food source is chosen with the probability which is proportioal to its quality*/
/*Different schemes can be used to calculate the probability values*/
/*For example prob(i)=fitness(i)/sum(fitness)*/
/*or in a way used in the metot below prob(i)=a*fitness(i)/max(fitness)+b*/
/*probability values are calculated by using fitness values and normalized by dividing maximum fitness value*/
void CalculateProbabilities(thread_data* thread_data)
{
    int i;
    double maxfit;
    maxfit = fitness[thread_data->startIdx];
    for (i = thread_data->startIdx; i < thread_data->endIdx; i++)
    {
        if (fitness[i] > maxfit)
            maxfit = fitness[i];
    }

    for (i = thread_data->startIdx; i < thread_data->endIdx; i++)
    {
        prob[i] = (0.9 * (fitness[i] / maxfit)) + 0.1;
    }
}

void SendOnlookerBees(thread_data* thread_data)
{

    int i, j, t;
    i = thread_data->startIdx;
    t = thread_data->startIdx;
    unsigned int *seed = &thread_data->seed;
    /*onlooker Bee Phase*/
    while (t < thread_data->endIdx)
    {
        double r = ((double)rand_r(seed) / ((double)(RAND_MAX) + (double)(1)));
        if (r < prob[i]) /*choose a food source depending on its probability to be chosen*/
        {
            t++;

            /*The parameter to be changed is determined randomly*/
            r = ((double)rand_r(seed) / ((double)(RAND_MAX) + (double)(1)));
            int param2change = (int)(r * D);

            /*A randomly chosen solution is used in producing a mutant solution of the solution i*/
            r = ((double)rand_r(seed) / ((double)(RAND_MAX) + (double)(1)));
            int neighbour = thread_data->startIdx + (int)(r * SwarmFoodNumber);

            /*Randomly selected solution must be different from the solution i*/
            while (neighbour == i)
            {
                r = ((double)rand_r(seed) / ((double)(RAND_MAX) + (double)(1)));
                neighbour = thread_data->startIdx + (int)(r * SwarmFoodNumber);
            }
            for (j = 0; j < D; j++)
                solution[thread_data->threadIdx][j] = Foods[i][j];

            /*v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) */
            r = ((double)rand_r(seed) / ((double)(RAND_MAX) + (double)(1)));
            solution[thread_data->threadIdx][param2change] = Foods[i][param2change] + (Foods[i][param2change] - Foods[neighbour][param2change]) * (r - 0.5) * 2;

            /*if generated parameter value is out of boundaries, it is shifted onto the boundaries*/
            if (solution[thread_data->threadIdx][param2change] < lb)
                solution[thread_data->threadIdx][param2change] = lb;
            if (solution[thread_data->threadIdx][param2change] > ub)
                solution[thread_data->threadIdx][param2change] = ub;
            double ObjValSol = function(solution[thread_data->threadIdx]);
            double FitnessSol = CalculateFitness(ObjValSol);

            /*a greedy selection is applied between the current solution i and its mutant*/
            if (FitnessSol > fitness[i])
            {
                /*If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i*/
                trial[i] = 0;
                for (j = 0; j < D; j++)
                    Foods[i][j] = solution[thread_data->threadIdx][j];
                f[i] = ObjValSol;
                fitness[i] = FitnessSol;
            }
            else
            { /*if the solution i can not be improved, increase its trial counter*/
                trial[i] = trial[i] + 1;
            }
        } /*if */
        i++;
        if (i == thread_data->endIdx)
            i = thread_data->startIdx;
    } /*while*/

    /*end of onlooker bee phase     */
}

/*determine the food sources whose trial counter exceeds the "limit" value. In Basic ABC, only one scout is allowed to occur in each cycle*/
void SendScoutBees(thread_data* thread_data)
{
    int maxtrialindex, i;
    maxtrialindex = 0;
    for (i = thread_data->startIdx; i < thread_data->endIdx; i++)
    {
        if (trial[i] > trial[maxtrialindex])
            maxtrialindex = i;
    }
    if (trial[maxtrialindex] >= limit)
    {
        init(maxtrialindex, &thread_data->seed, thread_data->threadIdx);
    }
}

/*Main program of the ABC algorithm*/
void *ABC(void *data)
{
    thread_data * ABC_data;
    ABC_data = (thread_data *) data;

    int iter; //, run, j;
    // double mean;
    // mean = 0;

    for (iter = 0; iter < maxCycle; iter++)
    {
        SendEmployedBees(ABC_data);
        CalculateProbabilities(ABC_data);
        SendOnlookerBees(ABC_data);
        MemorizeBestSource(ABC_data);
        SendScoutBees(ABC_data);
    }
    // for (j = 0; j < D; j++)
    // {
    //     printf("GlobalParam[%d]: %f\n", j + 1, GlobalParams[j]);
    // }
    // printf("%d. run: %e \n", run + 1, GlobalMin);
    // GlobalMins[run] = GlobalMin;
    // mean = mean + GlobalMin;

    return NULL;
}

/*

    Main program

*/
int main() 
{
    printf("Running ABC paralelly on CPU using %d processors and Colony size of %d\n", P, NP);

    struct timespec start, stop; 
	double t;

    double mean;
    mean = 0;

    // measure the start time here
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

    for (int run = 0; run < runtime; run++)
    {

        pthread_t threads[P];
        thread_data thread_data_array[P];

        int i, rc;
        for (i = 0; i < P; i++)
        {
            thread_data_array[i].threadIdx = i;
            thread_data_array[i].startIdx = i * SwarmFoodNumber;
            thread_data_array[i].endIdx =  thread_data_array[i].startIdx + SwarmFoodNumber;
            thread_data_array[i].seed = time(NULL) + i; // Initialize seed
        }

        initial(thread_data_array);
        MemorizeBestSourceForAll();

        for (i = 0; i < P; i++)
        {
            rc = pthread_create(&threads[i], NULL, ABC, (void *) &thread_data_array[i]);
            if (rc) 
            {
                printf("ERROR; return code form pthread_create() is %d\n", rc); 
                exit(-1);
            }
        }
        

        for (i = 0; i < P; i++)
        {
            rc = pthread_join(threads[i], NULL);
            if (rc) {
                printf("ERROR; return code from pthread_create() is %d\n", rc); 
                exit(-1);
            }
        }

        // Aggregation of results to find the best overall solution
        double overallBest = GlobalMin[0];
        int bestSwarmIndex = 0;
        for (int i = 1; i < P; i++) {
            if (GlobalMin[i] < overallBest) {
                overallBest = GlobalMin[i];
                bestSwarmIndex = i;
            }
        }

        // Output the best overall solution
        // printf("Best solution found by swarm %d: %e\n", bestSwarmIndex + 1, overallBest);
        // printf("Parameters of the best solution:\n");
        // for (int j = 0; j < D; j++) {
        //     printf("Param[%d]: %f\n", j + 1, GlobalParams[bestSwarmIndex][j]);
        // }

        // for (j = 0; j < D; j++)
        // {
        //     printf("GlobalParam[%d]: %f\n", j + 1, GlobalParams[j]);
        // }

        printf("%d. run: %e \n", run + 1, overallBest);
        GlobalMins[run] = overallBest;
        mean = mean + overallBest;
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

    return 0;

}




/*

    Benchmark Functions

*/

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