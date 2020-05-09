#include<stdio.h>
#include<omp.h>
#define INTERVALS 1000000000 

double f(double x) {
	return (4.0 / (1.0 + x*x));
}

double CalcPi (long int n) {
	const double dx = 1.0 / (double) n;
	double Sum = 0.0;
	double x;
	long int i;
	#pragma omp parallel for private(x,i) reduction(+:Sum)
	for (i = 0; i < n; i++) {
		x = dx * ((double)i + 0.5);
		Sum += f(x);
	}
	return dx * Sum;
}

int main(void){
	printf("Approx value of pi is: %.20lf", CalcPi(INTERVALS));
	return 0;
}
