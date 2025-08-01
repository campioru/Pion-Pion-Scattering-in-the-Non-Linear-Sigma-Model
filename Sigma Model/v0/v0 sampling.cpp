#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
using namespace std;

auto r = mt19937 {123};
auto norm = normal_distribution<double>{0, 1./sqrt(2.)};
auto expo = exponential_distribution<double>{1.};
auto uni = uniform_real_distribution<double>{0., 1.};

void mean_std(const vector<int>& vec, double& mean, double& std)
{
  mean = 0.;
  for (int i = 0; i < vec.size(); i ++) mean += double(vec[i]);
  mean /= vec.size();
  std = 0.;
  for (int i = 0; i < vec.size(); i ++) std += pow(double(vec[i]) - mean, 2.);
  std = pow(std / (vec.size() * (vec.size() - 1.)), .5);
}

double random_v0(const double& rho, int& rejections)
{
  double A, U, v0;
  rejections = -1;
  do
  {
    A = norm(r);
    U = uni(r);
    v0 = 1. - (A*A + expo(r)) / rho;
    rejections ++;
  }
  while (2*U*U > v0 + 1.);
  return v0;
}

int main()
{
  vector<double> rhos = {.01, pow(10., -1.5), .1, pow(10., -.5), 1., pow(10., .5), 10., pow(10., 1.5), 100.};
  int N = 1000000;
  vector<int> rejs(N);
  ofstream file;
  file.open("v0 sampling.csv");
  file << N << " iterations\n\n";
  for (int ρ = 0; ρ < rhos.size(); ρ ++)
  {
    file << "ρ," << rhos[ρ] << "\n";
    for (int n = 0; n < N-1; n ++) file << random_v0(rhos[ρ], rejs[n]) << ",";
    file << random_v0(rhos[ρ], rejs[N-1]) << "\n";
    double mean, std;
    mean_std(rejs, mean, std);
    file << "Rejections per step," << mean << "\nError," << std << "\n";
    cout << rhos[ρ] << "\n";
  }
  file.close();

  return 0;
}
