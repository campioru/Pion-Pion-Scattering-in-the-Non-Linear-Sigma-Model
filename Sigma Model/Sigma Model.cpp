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
    int d_, L_total_, T_, Δ_;
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
    Field(const vector<int>& L, const double& β, const double& λ) : d_(L.size()), L_total_(1), T_(L.back()), Δ_(T_/2 + 1), L_(L), β_(β), λ_(λ)
    {
      for (int d = 0; d < d_; d ++) L_total_ *= L_[d];
      for (int x = 0; x < L_total_; x ++) data_.push_back(SU2({ 1., 0., 0., 0. }));
    }

    SU2& operator() (const int& x)       { return data_[x]; }
    SU2  operator() (const int& x) const { return data_[x]; }
    int d() const { return d_; }
    int L_total() const { return L_total_; }
    int T() const { return T_; }
    int Delta() const { return Δ_; }
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

void mse_err(const vector<double>& vec, const double& mean, double& mse, double& err)
{
  int N = vec.size();

  double var = 0.;
  for (int i = 0; i < N; i ++) var += pow(vec[i] - mean, 2.);
  var /= double(N-1);

  double result = 0.;
  for (int i = 0; i < N; i ++) result += pow(vec[i] - mean, 4.);

  mse = var / double(N);
  err = pow(((result/double(N)) - pow(var, 2.)*double(N-3)/double(N-1)) / pow(double(N), 3.), .5);
}

void mse_err(const vector<int>& vec, const double& mean, double& mse, double& err)
{
  int N = vec.size();

  double var = 0.;
  for (int i = 0; i < N; i ++) var += pow(vec[i] - mean, 2.);
  var /= double(N-1);

  double result = 0.;
  for (int i = 0; i < N; i ++) result += pow(vec[i] - mean, 4.);

  mse = var / double(N);
  err = pow(((result/double(N)) - pow(var, 2.)*double(N-3)/double(N-1)) / pow(double(N), 3.), .5);
}


double random_v0(const double& rho, int& rejections)
{
  double A, U, v0;
  rejections --;
  do
  {
    A = n_dist(r);
    U = u_dist(r);
    v0 = 1. - (A*A + e_dist(r)) / rho;
    rejections ++;
  }
  while (2*U*U > v0 + 1.);
  return v0;
}

SU2 random_SU2(const double& ρ, int& rejections)
{
  valarray<double> v(4);
  v[0] = random_v0(ρ, rejections);
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

SU2 GibbsStep(Field& σ, vector<int>& coord, int& rejections)
{
  valarray<double> Σ = σ.Sigma(coord);
  double sqrtdet = 0.;
  for (int i = 0; i < 4; i ++) sqrtdet += Σ[i] * Σ[i];
  sqrtdet = pow(sqrtdet, .5);
  return random_SU2(σ.beta() * sqrtdet, rejections) * SU2(Σ / sqrtdet);
}

void GibbsCorrelation(Field& σ, const int& N, const int& disc, vector<vector<double>>& c_means, vector<vector<vector<double>>>& c_mses, vector<vector<vector<double>>>& c_mse_errs, double& rej_mean, vector<double>& rej_mses, vector<double>& rej_mse_errs, vector<vector<double>>& am_means, vector<vector<double>>& am_mses)
{
  auto start = chrono::steady_clock::now();
  vector<int> coord;
  int disc_rej = 0;
  for (int n = 0; n < disc; n ++)
  {
    coord = coord_init(σ.d());
    for (int x = 0; x < σ.L_total(); x ++)
    {
      σ(x) = GibbsStep(σ, coord, disc_rej);
      coord_update(coord, σ);
    }
  }
  chrono::duration<double> time = chrono::steady_clock::now() - start;
  cout << "Discarding: " << time.count() << " seconds\n";

  start = chrono::steady_clock::now();
  vector<vector<double>> Φs(σ.T());
  for (int t = 0; t < σ.T(); t ++) Φs[t] = vector<double>(4);
  for (int i = 0; i < 5; i ++)
  {
    for (int δ = 0; δ < σ.Delta(); δ ++) c_means[i][δ] = 0.;
  }
  rej_mean = 0.;
  vector<vector<vector<double>>> cs(N);
  vector<int> rs(N);

  for (int n = 0; n < N; n ++)
  {
    for (int t = 0; t < σ.T(); t ++)
    {
      for (int i = 0; i < 4; i ++) Φs[t][i] = 0.;
    }

    coord = coord_init(σ.d());
    rs[n] = 0;
    for (int x = 0; x < σ.L_total(); x ++)
    {
      σ(x) = GibbsStep(σ, coord, rs[n]);
      for (int i = 0; i < 4; i ++) Φs[coord.back()][i] += σ(x)[i];
      coord_update(coord, σ);
    }

    cs[n] = vector<vector<double>>(5);
    for (int i = 0; i < 5; i ++)
    {
      cs[n][i] = vector<double>(σ.Delta());
      for (int δ = 0; δ < σ.Delta(); δ ++) cs[n][i][δ] = 0.;
    }

    for (int δ = 0; δ < σ.Delta(); δ ++)
    {
      for (int i = 0; i < 4; i ++)
      {
        for (int t = 0; t < σ.T(); t ++) cs[n][i][δ] += Φs[t][i] * Φs[(t+δ) % σ.T()][i];
      }
      for (int i = 1; i < 4; i ++) cs[n][4][δ] += cs[n][i][δ];
      cs[n][4][δ] /= 3.;
      for (int i = 0; i < 5; i ++) c_means[i][δ] += cs[n][i][δ];
    }
    rej_mean += rs[n];
  }
  time = chrono::steady_clock::now() - start;
  cout << "Simulating: " << time.count() << " seconds\n";

  start = chrono::steady_clock::now();
  int K = int(.5 + log2(double(N)));
  for (int i = 0; i < 5; i ++)
  {
    for (int δ = 0; δ < σ.Delta(); δ ++)
    {
      c_means[i][δ] /= N;
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
    for (int n = 0; n < N; n ++) cs[n][i].resize(3);
  }
  time = chrono::steady_clock::now() - start;
  cout << "Correlation binning: " << time.count() << " seconds\n";

  start = chrono::steady_clock::now();
  rej_mean /= N;
  int M = N;
  for (int k = 0; k < K; k ++)
  {
    mse_err(rs, rej_mean, rej_mses[k], rej_mse_errs[k]);
    M /= 2;
    for (int n = 0; n < M; n ++) rs[n] = .5 * (rs[2*n] + rs[2*n + 1]);
    rs.resize(M);
  }
  time = chrono::steady_clock::now() - start;
  cout << "Rejection binning: " << time.count() << " seconds\n";

  start = chrono::steady_clock::now();
  M = N;
  for (int k = 0; k < K; k ++)
  {
    vector<double> cj(3);
    vector<double> amj(M);
    for (int i = 0; i < 5; i ++)
    {
      am_means[i][k] = 0.;
      for (int n = 0; n < M; n ++)
      {
        for (int δ = 0; δ <= 2; δ ++) cj[δ] = ((M*c_means[i][δ]) - cs[n][i][δ]) / (M - 1.);
        amj[n] = log((cj[0] - cj[2] + sqrt(pow(cj[0] - cj[2], 2.) - 4.*(cj[1]-cj[2])*(cj[0]-cj[1]))) / (2.*(cj[1] - cj[2])));
        am_means[i][k] += amj[n];
      }
      am_means[i][k] /= M;

      am_mses[i][k] = 0.;
      for (int n = 0; n < M; n ++) am_mses[i][k] += pow(am_means[i][k] - amj[n], 2.);
      am_mses[i][k] *= (M - 1.) / double(M);
    }
    M /= 2;
    for (int n = 0; n < M; n ++)
    {
      for (int i = 0; i < 5; i ++)
      {
        for (int δ = 0; δ <= 2; δ ++) cs[n][i][δ] = .5 * (cs[2*n][i][δ] + cs[2*n + 1][i][δ]);
      }
    }
    cs.resize(M);
  }
  time = chrono::steady_clock::now() - start;
  cout << "Mass jackknifing: " << time.count() << " seconds\n";
}
