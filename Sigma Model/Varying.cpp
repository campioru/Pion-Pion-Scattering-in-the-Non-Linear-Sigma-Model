#include <iostream>
#include <fstream>
#include <vector>
#include <valarray>
#include <cmath>
#include <random>
#include <complex>
#include <chrono>
using namespace std;

auto r = mt19937 {123};
auto n_dist = normal_distribution<double>{0, 1./sqrt(2.)};
auto e_dist = exponential_distribution<double>{1.};
auto u_dist = uniform_real_distribution<double>{0., 1.};
auto n1_dist = uniform_real_distribution<double>{-1., 1.};
static const double π = 3.14159265358979323846264338327950;
auto θ_dist = uniform_real_distribution<double>{0., 2.*π};


class SU2
{
  private:
    valarray<double> u_;
  public:
    SU2(const valarray<double>& u) : u_(u) {}
    double  operator[] (const int& i) const { return u_[i]; }
    valarray<double> comp() const { return u_; }
    SU2 operator* (const SU2& v) const { return SU2({
      u_[0]*v[0] - u_[1]*v[1] - u_[2]*v[2] - u_[3]*v[3],
      u_[0]*v[1] + u_[1]*v[0] - u_[2]*v[3] + u_[3]*v[2],
      u_[0]*v[2] + u_[1]*v[3] + u_[2]*v[0] - u_[3]*v[1],
      u_[0]*v[3] - u_[1]*v[2] + u_[2]*v[1] + u_[3]*v[0]}); }
};

class Field
{
  private:
    int d_, L_total_, t_;
    vector<int> L_;
    double β_, λ_;
    vector<SU2> data_;
    int index_(const vector<int>& coord) const
    {
      int index = coord[0], Π = L_[0];
      for (int i = 1; i < d_; i ++)
      {
        index += coord[i] * Π;
        Π *= L_[i];
      }
      return index;
    }
  public:
    Field(const vector<int>& L, const double& β, const double& λ) : d_(L.size()), L_total_(1), t_(L.back()), L_(L), β_(β), λ_(λ)
    {
      for (int d = 0; d < d_; d ++) L_total_ *= L_[d];
      for (int x = 0; x < L_total_; x ++) data_.push_back(SU2({ 1., 0., 0., 0. }));
    }

    SU2& operator() (const int& x)       { return data_[x]; }
    SU2  operator() (const int& x) const { return data_[x]; }
    int d() const { return d_; }
    int L_total() const { return L_total_; }
    int t() const { return t_; }
    int L(const int& d) const { return L_[d]; }
    double beta() const { return β_; }
    valarray<double> Sigma(vector<int>& coord) const
    {
      valarray<double> Σ = { λ_, 0., 0., 0. };
      for (int i = 0; i < d_; i ++)
      {
        coord[i] = (coord[i]+1) % L_[i];
        Σ += data_[index_(coord)].comp();
        coord[i] = (coord[i]-2+L_[i]) % L_[i];
        Σ += data_[index_(coord)].comp();
        coord[i] = (coord[i]+1) % L_[i];
      }
      return Σ;
    }
};


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

vector<int> coord_init(const int& length)
{
  vector<int> coord(length);
  for (int d = 0; d < length; d ++) coord[d] = 0;
  return coord;
}

void coord_update(vector<int>& coord, const Field& φ)
{
  coord[0] += 1;
    for (int d = 0; d < φ.d()-1; d ++)
    {
      if (coord[d] == φ.L(d))
      {
        coord[d] = 0;
        coord[d+1] += 1;
      }
      else break;
    }
}


SU2 random_SU2(const double& ρ, int& trials)
{
  valarray<double> v(4);
  {
    double U;
    do
    {
      double A = n_dist(r);
      U = u_dist(r);
      v[0] = 1. - (A*A + e_dist(r)) / ρ;
      trials ++;
    }
    while (2*U*U > v[0] + 1.);
  }
  v[1] = n1_dist(r);
  {
    double θ = θ_dist(r);
    v[2] = cos(θ);
    v[3] = sin(θ);
  }
  {
    double sqrt = pow(1. - v[1]*v[1], .5);
    for (int i = 2; i < 4; i ++) v[i] *= sqrt;
    sqrt = pow(1. - v[0]*v[0], .5);
    for (int i = 1; i < 4; i ++) v[i] *= sqrt;
  }
  return SU2(v);
}

SU2 GibbsStep(Field& σ, vector<int>& coord, int& trials)
{
  valarray<double> Σ = σ.Sigma(coord);
  double sqrtdet = 0.;
  for (int i = 0; i < 4; i ++) sqrtdet += Σ[i] * Σ[i];
  sqrtdet = pow(sqrtdet, .5);
  return random_SU2(σ.beta() * sqrtdet, trials) * SU2(Σ / sqrtdet);
}

void GibbsCorrelation(Field& σ, const int& N, const int& disc, vector<vector<double>>& c_means, vector<vector<vector<double>>>& c_mses, vector<vector<vector<double>>>& c_mse_errs, double& t_mean, vector<double>& t_mses, vector<double>& t_mse_errs)
{
  auto start = chrono::steady_clock::now();
  int disc_trials = 0;
  for (int n = 0; n < disc; n ++)
  {
    vector<int> coord = coord_init(σ.d());
    for (int x = 0; x < σ.L_total(); x ++)
    {
      σ(x) = GibbsStep(σ, coord, disc_trials);
      coord_update(coord, σ);
    }
  }
  chrono::duration<double> time = chrono::steady_clock::now() - start;
  cout << "Discarding: " << time.count() << " seconds\n";

  start = chrono::steady_clock::now();
  vector<vector<double>> Φs(σ.t());
  for (int t = 0; t < σ.t(); t ++) Φs[t] = vector<double>(4);
  for (int i = 0; i < 2; i ++)
  {
    c_means[i][0] = 1.;
    for (int δ = 1; δ <= σ.t()/2; δ ++) c_means[i][δ] = 0.;
  }
  t_mean = 0.;
  vector<vector<vector<double>>> cs(N);
  vector<double> ts(N);

  for (int n = 0; n < N; n ++)
  {
    int trials = 0;
    for (int t = 0; t < σ.t(); t ++)
    {
      for (int i = 0; i < 4; i ++) Φs[t][i] = 0.;
    }

    vector<int> coord = coord_init(σ.d());
    for (int x = 0; x < σ.L_total(); x ++)
    {
      σ(x) = GibbsStep(σ, coord, trials);
      for (int i = 0; i < 4; i ++) Φs[coord.back()][i] += σ(x)[i];
      coord_update(coord, σ);
    }

    cs[n] = vector<vector<double>>(2);
    for (int i = 0; i < 2; i ++)
    {
      cs[n][i] = vector<double>(σ.t()/2 + 1);
      for (int δ = 0; δ <= σ.t()/2; δ ++) cs[n][i][δ] = 0.;
    }

    for (int δ = 0; δ <= σ.t()/2; δ ++)
    {
      for (int t = 0; t < σ.t(); t ++)
      {
        cs[n][0][δ] += Φs[t][0] * Φs[(t+δ) % σ.t()][0];
        for (int i = 1; i < 4; i ++) cs[n][1][δ] += Φs[t][i] * Φs[(t+δ) % σ.t()][i];
      }
    }

    for (int i = 0; i < 2; i ++)
    {
      for (int δ = 1; δ <= σ.t()/2; δ ++)
      {
        cs[n][i][δ] /= cs[n][i][0];
        c_means[i][δ] += cs[n][i][δ];
      }
      cs[n][i][0] = 1.;
    }
    ts[n] = double(trials) / σ.L_total();
    t_mean += ts[n];
  }
  time = chrono::steady_clock::now() - start;
  cout << "Simulating: " << time.count() << " seconds\n";

  start = chrono::steady_clock::now();
  int K = int(.5 + log2(double(N)));
  for (int i = 0; i < 2; i ++)
  {
    for (int k = 0; k < K; k ++)
    {
      c_mses[i][0][k] = 0.;
      c_mse_errs[i][0][k] = 0.;
    }
    for (int δ = 1; δ <= σ.t()/2; δ ++)
    {
      c_means[i][δ] /= double(N);

      vector<double> c(N);
      for (int n = 0; n < N; n ++) c[n] = cs[n][i][δ];
      int M = N;
      for (int k = 0; k < K; k ++)
      {
        mse_err(c, c_means[i][δ], c_mses[i][δ][k], c_mse_errs[i][δ][k]);
        M /= 2;
        for (int n = 0; n < M; n ++) c[n] = .5 * (c[2*n] + c[2*n + 1]);
        c.resize(M);
      }
    }
  }

  t_mean /= double(N);
  int M = N;
  for (int k = 0; k < K; k ++)
  {
    mse_err(ts, t_mean, t_mses[k], t_mse_errs[k]);
    M /= 2;
    for (int n = 0; n < M; n ++) ts[n] = .5 * (ts[2*n] + ts[2*n + 1]);
    ts.resize(M);
  }

  time = chrono::steady_clock::now() - start;
  cout << "Binning: " << time.count() << " seconds\n";
}


int main()
{
  return 0;
}