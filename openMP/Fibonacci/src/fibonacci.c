#include<stdio.h>
#include<omp.h>
#define N 34

int serialfib(int n) {
	if (n < 2) 
		return n;

	int x = serialfib(n - 1);
	int y = serialfib(n - 2);
	return x+y;
}

int parallelfib(int n) {
	if (n < 2) 
		return n;
	//Skip openMP overhead if n <= 30
	if (n <= 30)
		return serialfib(n);
	int x, y;
	#pragma omp task shared(x) 
	{
		x = parallelfib(n - 1);
	}
	#pragma omp task shared(y)
	{
		y = parallelfib(n - 2);
	}
	#pragma omp taskwait
	return x+y;
}

int main(void) {
	int nth_term;
	#pragma omp parallel
	{
		#pragma omp single
		{
			nth_term = parallelfib(N);
		}
	}
	printf("The %d term of fibonnaci series is: %d\n", N, nth_term);
	return 0;
}
