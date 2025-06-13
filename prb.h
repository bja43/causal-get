#ifndef PRB_H_
#define PRB_H_

#include <stdlib.h>
#include <assert.h>
#include <math.h>

double prb_rand(void);
double prb_randn(void);

double prb_norm_cdf(double x);
double prb_norm_invcdf(double x);

void prb_shuffle(int *arr, size_t size);

// float pi = atan2f(1, 1) * 4

#endif // PRB_H_

#ifdef PRB_IMPLEMENTATION

double prb_rand(void)
{
  return (double) rand() / RAND_MAX;
}

double prb_randn(void)
{
  double x = prb_rand();

  return prb_norm_invcdf(x);
}

// A Logistic Approximation to the Cumulative Normal Distribution
// Bowling S. R., Khasawneh M. T., Kaewkuekool S. and Cho B. R.
double prb_norm_cdf(double x)
{
  double a = -1.5976f;
  double b = 0.07056f;

  return 1.0 / (1.0 + exp(a * x - b * pow(x, 3.0)));
}

// Very Simply Explicitly Invertible Approximations
// of Normal Cumulative and Normal Quantile Function
// Soranzo, A. and Epure, E.
double prb_norm_invcdf(double x)
{
  assert(0.0 < x && x < 1.0);

  double a = 2.69282508;    // 10 / log(41)
  double b = -0.11857259;   // log(log(2)) / log(22)
  double c = 3.09104245;    // log(22)

  if (x < 0.5) {
    x = 1.0 - x;
    a *= -1.0;
  }
    
  return a * log(1.0 + b - log(-log(x)) / c);
}

// Fisher-Yates algorithm
void prb_shuffle(int *arr, size_t size)
{
  if (size < 2) return;
  
  for (size_t i = size - 1; i > 0; i--) {
    size_t j = rand() % (i + 1);   // randint(i + 1)
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

#endif // PRB_IMPLEMENTATION
