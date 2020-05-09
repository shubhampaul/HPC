#include <stdio.h>
#include <omp.h>

int main (void) {
    int i = 10;
    #pragma omp parallel private(i) num_threads(6)
    {
	    printf("thread %d: i = %d\n", omp_get_thread_num(), 
			    omp_get_num_threads());	
    }
    printf("i = %d\n", i);

    return 0;
}
