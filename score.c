#include <stdio.h>
#include <stdlib.h>

float addf(float a, float b)
{
  return a + b;
}

int main()
{
  float (*func)(float, float);
  func = addf;

  float a = 1.2f;
  float b = 3.4f;

  printf("%4.2f + %4.2f = %4.2f\n", a, b, func(a, b));

  return 0;
}
