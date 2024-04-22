/* 
This program will numerically compute the integral of
                  4/(1+x*x)				  
from 0 to 1.  The value of this integral is pi. 
It uses the timer from the OpenMP runtime library
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define NUM_THREADS 12
double compute_pi_false_sharing(double step);
double compute_pi_race_condition(double step);
double compute_pi_none(double step);
double compute_pi_serial(double step);
void usage(char prog_name[]);
static long num_steps = 1000000;

//Adding a strcut to ensure that every thread is placed at the start of different cache lines
struct padded_sum{
	double value;
	char padding[64-sizeof(double)];
};

double speedup(double serial, double parallel){

	return serial/parallel;

}

int main (int argc, char **argv)
{
	double start_time, run_time=0, pi, step;
	int iter;
	double time_serial, time_parallel_false_sharing, time_parallel_race_condition, time_parallel_none;
	double speedupCalc;
	if (argc != 2) {
		usage(argv[0]);
		exit (-1);
	}
	
	iter=atoi(argv[1]);
	step = 1.0/(double)num_steps;
	//serial code
	for(int i=0; i<iter; i++){
		start_time = omp_get_wtime();
		pi=compute_pi_serial(step);
		run_time += omp_get_wtime() - start_time;
	}
	time_serial = run_time/iter;
	printf("\nSequential : pi with %ld steps is %f in %f seconds\n",num_steps,pi,run_time/iter);
	//parallel code with false sharing
	run_time = 0;
	for(int i=0; i<iter; i++){
		start_time = omp_get_wtime();
		pi=compute_pi_false_sharing(step);
		run_time += omp_get_wtime() - start_time;
	}
	speedupCalc = speedup(time_serial, run_time/iter);
	printf("\nParallel with false sharing : pi with %ld steps is %f in %f seconds using %d threads\nSpeedup: %f\n",num_steps,pi,run_time/iter,NUM_THREADS, speedupCalc);
	//parallel code with race condition
	run_time = 0;
	for(int i=0; i<iter; i++){
		start_time = omp_get_wtime();
		pi=compute_pi_race_condition(step);
		run_time += omp_get_wtime() - start_time;
    }
		speedupCalc = speedup(time_serial, run_time/iter);
        printf("\nParallel with race condition : pi with %ld steps is %f in %f seconds using %d threads\nSpeedup: %f\n",num_steps,pi,run_time/iter,NUM_THREADS, speedupCalc);
	//parallel code with no race condition and false sharing
	run_time = 0;
	for(int i=0; i<iter; i++){
                start_time = omp_get_wtime();
                pi=compute_pi_none(step);
                run_time += omp_get_wtime() - start_time;
        }
		speedupCalc = speedup(time_serial, run_time/iter);
        printf("\nParallel with no race condition and false sharing : pi with %ld steps is %f in %f seconds using %d threads\nSpeedup: %f\n",num_steps,pi,run_time/iter,NUM_THREADS, speedupCalc);
}	  

double compute_pi_serial(double step){
	int i;
	double x, pi, sum = 0.0;
	for (i=0;i<num_steps; i++){
		x = ( i +0.5) * step ;
		sum = sum + 4.0/(1.0+ x * x ) ;
	}
	pi = step * sum;
	return pi;
}

double compute_pi_false_sharing(double step){
	int nthreads;
	double pi = 0.0, sum[NUM_THREADS];
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel
	{
		int i, id, tthreads;
		double x;
		tthreads = omp_get_num_threads();
		id= omp_get_thread_num();
		if(id ==0)
		{
			nthreads=tthreads;
		}
		for (i=id,sum[id]=0.0;i<num_steps;i=i+tthreads){
			x = ( i +0.5) * step ;
			sum[id] = sum[id] + 4.0/(1.0+ x * x ) ;
		}
	}
	
	for ( int i =0; i < nthreads ; i ++){
		pi += step * sum [ i ];
	}
	return pi;
}

double compute_pi_race_condition(double step){

	int i , tthreads , id ;
	double pi = 0.0 , sum = 0.0 ,x; //x is shared here	
	#pragma omp parallel
	{
		tthreads = omp_get_num_threads () ;
		id = omp_get_thread_num () ;
		for (i = id ; i < num_steps ; i += tthreads ) {
			x = ( i +0.5) * step ;
			sum = sum + 4.0/(1.0+ x * x ) ;
		}
	}
	pi = step*sum;
	return pi;
}

double compute_pi_none(double step) {
    int nthreads;
    struct padded_sum sum[NUM_THREADS];
    double pi = 0.0;
    int i, id, tthreads;
    double x;

    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel private(i, x, id, tthreads)
    {
        id = omp_get_thread_num();
        tthreads = omp_get_num_threads();

        if (id == 0) {
            nthreads = tthreads;
        }

        // Initialize local sum for each thread
        sum[id].value = 0.0;

        // Loop unrolling by 4
        for (i = id; i < num_steps; i += tthreads * 4) {
            x = (i + 0.5) * step;
            sum[id].value += 4.0 / (1.0 + x * x);

            x = (i + 1 + 0.5) * step;
            sum[id].value += 4.0 / (1.0 + x * x);

            x = (i + 2 + 0.5) * step;
            sum[id].value += 4.0 / (1.0 + x * x);

            x = (i + 3 + 0.5) * step;
            sum[id].value += 4.0 / (1.0 + x * x);
        }
    }

    // Final accumulation
    #pragma omp parallel for reduction(+:pi)
    for (i = 0; i < nthreads; i++) {
        pi += step * sum[i].value;
    }

    return pi;
}

/*--------------------------------------------------------------------
 * Function:    usage
 * Purpose:     Print command line for function
 * In arg:      prog_name
 */
void usage(char prog_name[]) {
   fprintf(stderr, "usage:  %s <number of times to run>\n", prog_name);
} /* usage */



