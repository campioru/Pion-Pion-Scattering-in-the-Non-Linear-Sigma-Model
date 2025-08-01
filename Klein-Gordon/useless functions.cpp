#include <iostream>
#include <vector>
#include <cmath>
#include <random>
using namespace std;

auto r = mt19937 {123};
auto stand = uniform_real_distribution<double>{0., 1.};

void mse_err(const vector<double>& vec, const double& mean, double& mse, double& err)
{
  int i;
  int N = vec.size();

  double var = 0.;
  for (i = 0; i < N; i ++) var += pow(vec[i] - mean, 2.);
  var /= double(N-1);

  double result = 0.;
  for (i = 0; i < N; i ++) result += pow(vec[i] - mean, 4.);

  mse = var / double(N);
  err = pow(((result/double(N)) - pow(var, 2.)*double(N-3)/double(N-1)) / pow(double(N), 3.), .5);
}

class Field2d
{
  private:
    int L1_, L2_;
    double am_, κ_, κ2_;
    vector<double> data_;
    int index_(int x, int y) const { return x + L1_*y; }
  public:
    Field2d(int L1, int L2, double am) : L1_(L1), L2_(L2), am_(am), κ_(pow(4. + am_*am_, -.5)), κ2_(1. / (4. + am_*am_)), data_(L1_*L2_)
    {
      int L_total = L1_*L2_;
      for (int i = 0; i < L_total; i ++) data_[i] = 0.;
    }

    double& operator() (int x, int y)       { return data_[index_(x, y)]; }
    double  operator() (int x, int y) const { return data_[index_(x, y)]; }
    int L1() const { return L1_; }
    int L2() const { return L2_; }
    double am() const { return am_; }
    double kappa() const { return κ_; }
    double kappa2() const { return κ2_; }
};

void Gibbs(Field2d& φ)
{
  int i, j;
  double γ;
  for (j = 0; j < φ.L2(); j ++)
  {
    for (i = 0; i < φ.L1(); i ++)
    {
      γ = φ((i+1) % φ.L1(), j) + φ(i, (j+1) % φ.L2()) + φ((i-1+φ.L1()) % φ.L1(), j) + φ(i, (j-1+φ.L2()) % φ.L2());
      auto norm = normal_distribution<double>{φ.kappa2()*γ, φ.kappa()};
      φ(i, j) = norm(r);
    }
  }
}

void Metropolis(Field2d& φ, const auto& dist)
{
  int i, j;
  double curr, prop, κ2γ, diff;
  for (j = 0; j < φ.L2(); j ++)
  {
    for (i = 0; i < φ.L1(); i ++)
    {
      curr = φ(i, j);
      prop = curr + dist(r);
      κ2γ = φ.kappa2() * (φ((i+1) % φ.L1(), j) + φ(i, (j+1) % φ.L2()) + φ((i-1+φ.L1()) % φ.L1(), j) + φ(i, (j-1+φ.L2()) % φ.L2()));
      diff = pow(curr - κ2γ, 2.) - pow(prop - κ2γ, 2.);
      if (diff >= 0. || stand(r) < exp(diff / (2.*φ.kappa2()))) φ(i, j) = prop;
    }
  }
}

void GibbsCorrelation(Field2d& φ, const int& N, const int& disc, vector<vector<double>>& c_means, vector<vector<vector<double>>>& c_mses, vector<vector<vector<double>>>& c_mse_errs)
{
  for (int j = 0; j < φ.L2(); j ++)
  {
    for (int i = 0; i < φ.L1(); i ++) φ(i, j) = 0.;
  }

  for (int n = 0; n < disc; n ++)
  {
    Gibbs(φ);
    if (n % 100000 == 0) cout << n << "\n";
  }

  int K = int(.5 + log2(double(N)));

  vector<vector<double>> Φs(2);
  Φs[0] = vector<double>(φ.L1());
  Φs[1] = vector<double>(φ.L2());

  // FIX L -> L+1 FOR CS VECTOR AND LOOPS
  vector<vector<vector<double>>> cs(2);
  cs[0] = vector<vector<double>>(φ.L1());
  cs[1] = vector<vector<double>>(φ.L2());
  for (int i = 0; i < φ.L1(); i ++) cs[0][i] = vector<double>(N);
  for (int i = 0; i < φ.L2(); i ++) cs[1][i] = vector<double>(N);

  for (int i = 0; i < φ.L1(); i ++)
  {
    Φs[0][i] = 0.;
    for (int j = 0; j < N; j ++) cs[0][i][j] = 0.;
    for (int j = 0; j < K; j ++)
    {
      c_mses[0][i][j] = 0.;
      c_mse_errs[0][i][j] = 0.;
    }
  }
  for (int i = 0; i < φ.L1()+1; i ++) c_means[0][i] = 0.;
  for (int i = 0; i < φ.L2(); i ++)
  {
    Φs[1][i] = 0.;
    for (int j = 0; j < N; j ++) cs[1][i][j] = 0.;
    for (int j = 0; j < K; j ++)
    {
      c_mses[1][i][j] = 0.;
      c_mse_errs[1][i][j] = 0.;
    }
  }
  for (int i = 0; i < φ.L2()+1; i ++) c_means[1][i] = 0.;

  for (int n = 0; n < N; n ++)
  {
    Gibbs(φ);
    // SET PHIS TO BE 0 EACH ITERATION BEFORE SUMMING
    for (int i = 0; i < φ.L1(); i ++)
    {
      for (int j = 0; i < φ.L2(); j ++)
      {
        Φs[0][i] += φ(i, j);
        Φs[1][j] += φ(i, j);
      }
    }

    for (int i = 0; i < φ.L1(); i ++)
    {
      for (int j = 0; j <= φ.L1(); j ++)
      {
        cs[0][j][n] += Φs[0][i] * Φs[0][(i+j) % φ.L1()];
        c_means[0][j] += cs[0][j][n];
      }
    }
    for (int i = 0; i < φ.L2(); i ++)
    {
      for (int j = 0; j <= φ.L2(); j ++)
      {
        cs[1][j][n] += Φs[1][i] * Φs[1][(i+j) % φ.L2()];
        c_means[1][j] += cs[1][j][n];
      }
    }

    if (n % 100000 == 0) cout << n << "\n";
  }

  for (int i = 0; i < φ.L1()+1; i ++)
  {
    c_means[0][i] /= double(N);

    int bins = N;
    vector<double> c(bins);
    for (int n = 0; n < bins; n ++) c[n] = cs[0][i][n];
    mse_err(c, c_means[0][i], c_mses[0][i][0], c_mse_errs[0][i][0]);

    for (int j = 1; j < K; j ++)
    {
      bins /= 2;
      for (int n = 0; n < bins; n ++) c[n] = .5 * (c[2*n] + c[2*n + 1]);
      c.resize(bins);
      mse_err(c, c_means[0][i], c_mses[0][i][j], c_mse_errs[0][i][j]);
    }
  }
  for (int i = 0; i < φ.L2()+1; i ++)
  {
    c_means[1][i] /= double(N);

    int bins = N;
    vector<double> c(bins);
    for (int n = 0; n < bins; n ++) c[n] = cs[1][i][n];
    mse_err(c, c_means[1][i], c_mses[1][i][0], c_mse_errs[1][i][0]);

    for (int j = 1; j < K; j ++)
    {
      bins /= 2;
      for (int n = 0; n < bins; n ++) c[n] = .5 * (c[2*n] + c[2*n + 1]);
      c.resize(bins);
      mse_err(c, c_means[1][i], c_mses[1][i][j], c_mse_errs[1][i][j]);
    }
  }
}