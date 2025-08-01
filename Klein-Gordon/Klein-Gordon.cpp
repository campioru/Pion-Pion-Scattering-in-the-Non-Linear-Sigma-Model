#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
using namespace std;

auto r = mt19937 {123};

class Field
{
  private:
    int d_, L_total_;
    vector<int> L_;
    double am_, κ_;
    vector<double> data_;
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
    Field(const vector<int>& L, const double& am) : d_(L.size()), L_total_(1), L_(d_), am_(am), κ_(pow(2.*d_ + am_*am_, -.5)), data_(0)
    {
      L_ = L;
      for (int d = 0; d < d_; d ++) L_total_ *= L_[d];
      data_.resize(L_total_);
      for (int x = 0; x < L_total_; x ++) data_[x] = 0.;
    }

    double& operator() (const int& x)       { return data_[x]; }
    double  operator() (const int& x) const { return data_[x]; }
    int d() const { return d_; }
    int L_total() const { return L_total_; }
    int L(const int& d) const { return L_[d]; }
    double kappa() const { return κ_; }
    double gamma(vector<int>& coord) const
    {
      double γ = 0.;
      for (int i = 0; i < d_; i ++)
      {
        coord[i] = (coord[i]+1) % L_[i];
        γ += data_[index_(coord)];
        coord[i] = (coord[i]-2+L_[i]) % L_[i];
        γ += data_[index_(coord)];
        coord[i] = (coord[i]+1) % L_[i];
      }
      return γ;
    }
};

class FieldSquare
{
  private:
    int L_;
    double am_, κ_;
    vector<vector<double>> data_;
  public:
    FieldSquare(const int& L, const double& am) : L_(L), am_(am), κ_(pow(4. + am_*am_, -.5)), data_(L_)
    {
      for (int x = 0; x < L_; x ++)
      {
        data_[x] = vector<double>(L_);
        for (int y = 0; y < L_; y ++) data_[x][y] = 0.;
      }
    }

    double& operator() (const int& x, const int& y)       { return data_[x][y]; }
    double  operator() (const int& x, const int& y) const { return data_[x][y]; }
    int L() const { return L_; }
    double kappa() const { return κ_; }
    double gamma(const int& x, const int& y) const
    { return data_[(x-1+L_) % L_][y] + data_[(x+1) % L_][y] + data_[x][(y-1+L_) % L_] + data_[x][(y+1) % L_]; }
};

class FieldCube
{
  private:
    int L_;
    double am_, κ_;
    vector<vector<vector<double>>> data_;
  public:
    FieldCube(const int& L, const double& am) : L_(L), am_(am), κ_(pow(6. + am_*am_, -.5)), data_(L_)
    {
      for (int x = 0; x < L_; x ++)
      {
        data_[x] = vector<vector<double>>(L_);
        for (int y = 0; y < L_; y ++)
        {
          data_[x][y] = vector<double>(L_);
          for (int z = 0; z < L_; z ++) data_[x][y][z] = 0.;
        }
      }
    }

    double& operator() (const int& x, const int& y, const int& z)       { return data_[x][y][z]; }
    double  operator() (const int& x, const int& y, const int& z) const { return data_[x][y][z]; }
    int L() const { return L_; }
    double kappa() const { return κ_; }
    double gamma(const int& x, const int& y, const int& z) const
    { return data_[(x-1+L_) % L_][y][z] + data_[(x+1) % L_][y][z] + data_[x][(y-1+L_) % L_][z] + data_[x][(y+1) % L_][z] + data_[x][y][(z-1+L_) % L_] + data_[x][y][(z+1) % L_]; }
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

double normal_sample(const double& μ, const double& σ)
{
  auto norm = normal_distribution<double>{μ, σ};
  return norm(r);
}


void GibbsCorrelation(Field& φ, const int& N, const int& disc, vector<vector<double>>& c_means, vector<vector<vector<double>>>& c_mses, vector<vector<vector<double>>>& c_mse_errs)
{
  for (int n = 0; n < disc; n ++)
  {
    vector<int> coord = coord_init(φ.d());
    for (int x = 0; x < φ.L_total(); x ++)
    {
      φ(x) = normal_sample(φ.kappa()*φ.kappa()*φ.gamma(coord), φ.kappa());
      coord_update(coord, φ);
    }
    cout << n << "\n";
  }

  vector<vector<double>> Φs(φ.d());
  for (int d = 0; d < φ.d(); d ++)
  {
    Φs[d] = vector<double>(φ.L(d));
    for (int δ = 0; δ <= φ.L(d)/2; δ ++) c_means[d][δ] = 0.;
  }

  vector<vector<vector<double>>> cs(N);
  for (int n = 0; n < N; n ++)
  {
    for (int d = 0; d < φ.d(); d ++)
    {
      for (int x = 0; x < φ.L(d); x ++) Φs[d][x] = 0.;
    }

    vector<int> coord = coord_init(φ.d());
    for (int x = 0; x < φ.L_total(); x ++)
    {
      double site = normal_sample(φ.kappa()*φ.kappa()*φ.gamma(coord), φ.kappa());
      φ(x) = site;
      for (int d = 0; d < φ.d(); d ++) Φs[d][coord[d]] += site;
      coord_update(coord, φ);
    }

    cs[n] = vector<vector<double>>(φ.d());
    for (int d = 0; d < φ.d(); d ++)
    {
      cs[n][d] = vector<double>(φ.L(d)/2 + 1);
      for (int δ = 0; δ <= φ.L(d)/2; δ ++)
      {
        cs[n][d][δ] = 0.;
        for (int x = 0; x < φ.L(d); x ++) cs[n][d][δ] += Φs[d][x] * Φs[d][(x+δ) % φ.L(d)];
        c_means[d][δ] += cs[n][d][δ];
      }
    }
    cout << n << "\n";
  }

  int K = int(.5 + log2(double(N)));
  for (int d = 0; d < φ.d(); d ++)
  {
    for (int δ = 0; δ <= φ.L(d)/2; δ ++)
    {
      c_means[d][δ] /= double(N);

      vector<double> c(N);
      for (int n = 0; n < N; n ++) c[n] = cs[n][d][δ];
      int M = N;
      for (int k = 0; k < K; k ++)
      {
        mse_err(c, c_means[d][δ], c_mses[d][δ][k], c_mse_errs[d][δ][k]);
        M /= 2;
        for (int n = 0; n < M; n ++) c[n] = .5 * (c[2*n] + c[2*n + 1]);
        c.resize(M);
      }
    }
  }
}

void GibbsCorrelation(FieldSquare& φ, const int& N, const int& disc, vector<vector<double>>& c_means, vector<vector<vector<double>>>& c_mses, vector<vector<vector<double>>>& c_mse_errs)
{
  for (int n = 0; n < disc; n ++)
  {
    for (int y = 0; y < φ.L(); y ++)
    {
      for (int x = 0; x < φ.L(); x ++) φ(x, y) = normal_sample(φ.kappa()*φ.kappa()*φ.gamma(x, y), φ.kappa());
    }
    cout << n << "\n";
  }

  vector<vector<double>> Φs(2);
  for (int d = 0; d < 2; d ++)
  {
    Φs[d] = vector<double>(φ.L());
    for (int δ = 0; δ <= φ.L()/2; δ ++) c_means[d][δ] = 0.;
  }
  vector<vector<vector<double>>> cs(N);

  for (int n = 0; n < N; n ++)
  {
    for (int d = 0; d < 2; d ++)
    {
      for (int x = 0; x < φ.L(); x ++) Φs[d][x] = 0.;
    }

    for (int y = 0; y < φ.L(); y ++)
    {
      for (int x = 0; x < φ.L(); x ++)
      {
        double site = normal_sample(φ.kappa()*φ.kappa()*φ.gamma(x, y), φ.kappa());
        φ(x, y) = site;
        Φs[0][x] += site;
        Φs[1][y] += site;
      }
    }

    cs[n] = vector<vector<double>>(2);
    for (int d = 0; d < 2; d ++)
    {
      cs[n][d] = vector<double>(φ.L()/2 + 1);
      for (int δ = 0; δ <= φ.L()/2; δ ++)
      {
        cs[n][d][δ] = 0.;
        for (int x = 0; x < φ.L(); x ++) cs[n][d][δ] += Φs[d][x] * Φs[d][(x+δ) % φ.L()];
        c_means[d][δ] += cs[n][d][δ];
      }
    }
    cout << n << "\n";
  }

  int K = int(.5 + log2(double(N)));
  for (int d = 0; d < 2; d ++)
  {
    for (int δ = 0; δ <= φ.L()/2; δ ++)
    {
      c_means[d][δ] /= double(N);

      vector<double> c(N);
      for (int n = 0; n < N; n ++) c[n] = cs[n][d][δ];
      int M = N;
      for (int k = 0; k < K; k ++)
      {
        mse_err(c, c_means[d][δ], c_mses[d][δ][k], c_mse_errs[d][δ][k]);
        M /= 2;
        for (int n = 0; n < M; n ++) c[n] = .5 * (c[2*n] + c[2*n + 1]);
        c.resize(M);
      }
    }
  }
}

void GibbsCorrelation(FieldCube& φ, const int& N, const int& disc, vector<vector<double>>& c_means, vector<vector<vector<double>>>& c_mses, vector<vector<vector<double>>>& c_mse_errs)
{
  for (int n = 0; n < disc; n ++)
  {
    for (int z = 0; z < φ.L(); z ++)
    {
      for (int y = 0; y < φ.L(); y ++)
      {
        for (int x = 0; x < φ.L(); x ++) φ(x, y, z) = normal_sample(φ.kappa()*φ.kappa()*φ.gamma(x, y, z), φ.kappa());
      }
    }
    cout << n << "\n";
  }

  vector<vector<double>> Φs(3);
  for (int d = 0; d < 3; d ++)
  {
    Φs[d] = vector<double>(φ.L());
    for (int δ = 0; δ <= φ.L()/2; δ ++) c_means[d][δ] = 0.;
  }
  vector<vector<vector<double>>> cs(N);

  for (int n = 0; n < N; n ++)
  {
    for (int d = 0; d < 3; d ++)
    {
      for (int x = 0; x < φ.L(); x ++) Φs[d][x] = 0.;
    }

    for (int z = 0; z < φ.L(); z ++)
    {
      for (int y = 0; y < φ.L(); y ++)
      {
        for (int x = 0; x < φ.L(); x ++)
        {
          double site = normal_sample(φ.kappa()*φ.kappa()*φ.gamma(x, y, z), φ.kappa());
          φ(x, y, z) = site;
          Φs[0][x] += site;
          Φs[1][y] += site;
          Φs[2][z] += site;
        }
      }
    }

    cs[n] = vector<vector<double>>(3);
    for (int d = 0; d < 3; d ++)
    {
      cs[n][d] = vector<double>(φ.L()/2 + 1);
      for (int δ = 0; δ <= φ.L()/2; δ ++)
      {
        cs[n][d][δ] = 0.;
        for (int x = 0; x < φ.L(); x ++) cs[n][d][δ] += Φs[d][x] * Φs[d][(x+δ) % φ.L()];
        c_means[d][δ] += cs[n][d][δ];
      }
    }
    cout << n << "\n";
  }

  int K = int(.5 + log2(double(N)));
  for (int d = 0; d < 3; d ++)
  {
    for (int δ = 0; δ <= φ.L()/2; δ ++)
    {
      c_means[d][δ] /= double(N);

      vector<double> c(N);
      for (int n = 0; n < N; n ++) c[n] = cs[n][d][δ];
      int M = N;
      for (int k = 0; k < K; k ++)
      {
        mse_err(c, c_means[d][δ], c_mses[d][δ][k], c_mse_errs[d][δ][k]);
        M /= 2;
        for (int n = 0; n < M; n ++) c[n] = .5 * (c[2*n] + c[2*n + 1]);
        c.resize(M);
      }
    }
  }
}

void MetropolisCorrelation(Field& φ, const double& ε, const int& N, const int& disc, vector<vector<double>>& c_means, vector<vector<vector<double>>>& c_mses, vector<vector<vector<double>>>& c_mse_errs)
{
  auto dist = uniform_real_distribution<double>{-ε, ε};
  auto stand = uniform_real_distribution<double>{0., 1.};
  for (int n = 0; n < disc; n ++)
  {
    vector<int> coord = coord_init(φ.d());
    for (int x = 0; x < φ.L_total(); x ++)
    {
      double curr = φ(x);
      double prop = curr + dist(r);
      double κ2γ = φ.kappa() * φ.kappa() * φ.gamma(coord);
      double diff = pow(curr - κ2γ, 2.) - pow(prop - κ2γ, 2.);
      if (diff >= 0. || stand(r) < exp(diff / (2.*φ.kappa()*φ.kappa()))) φ(x) = prop;
      coord_update(coord, φ);
    }
    cout << n << "\n";
  }

  vector<vector<double>> Φs(φ.d());
  for (int d = 0; d < φ.d(); d ++)
  {
    Φs[d] = vector<double>(φ.L(d));
    for (int δ = 0; δ <= φ.L(d)/2; δ ++) c_means[d][δ] = 0.;
  }
  vector<vector<vector<double>>> cs(N);

  for (int n = 0; n < N; n ++)
  {
    for (int d = 0; d < φ.d(); d ++)
    {
      for (int x = 0; x < φ.L(d); x ++) Φs[d][x] = 0.;
    }

    vector<int> coord = coord_init(φ.d());
    for (int x = 0; x < φ.L_total(); x ++)
    {
      double curr = φ(x);
      double prop = curr + dist(r);
      double κ2γ = φ.kappa() * φ.kappa() * φ.gamma(coord);
      double diff = pow(curr - κ2γ, 2.) - pow(prop - κ2γ, 2.);
      if (diff >= 0. || stand(r) < exp(diff / (2.*φ.kappa()*φ.kappa())))
      {
        φ(x) = prop;
        for (int d = 0; d < φ.d(); d ++) Φs[d][coord[d]] += prop;
      }
      else
      {
        for (int d = 0; d < φ.d(); d ++) Φs[d][coord[d]] += curr;
      }
      coord_update(coord, φ);
    }

    cs[n] = vector<vector<double>>(φ.d());
    for (int d = 0; d < φ.d(); d ++)
    {
      cs[n][d] = vector<double>(φ.L(d)/2 + 1);
      for (int δ = 0; δ <= φ.L(d)/2; δ ++)
      {
        cs[n][d][δ] = 0.;
        for (int x = 0; x < φ.L(d); x ++) cs[n][d][δ] += Φs[d][x] * Φs[d][(x+δ) % φ.L(d)];
        c_means[d][δ] += cs[n][d][δ];
      }
    }
    cout << n << "\n";
  }

  int K = int(.5 + log2(double(N)));
  for (int d = 0; d < φ.d(); d ++)
  {
    for (int δ = 0; δ <= φ.L(d)/2; δ ++)
    {
      c_means[d][δ] /= double(N);

      vector<double> c(N);
      for (int n = 0; n < N; n ++) c[n] = cs[n][d][δ];
      int M = N;
      for (int k = 0; k < K; k ++)
      {
        mse_err(c, c_means[d][δ], c_mses[d][δ][k], c_mse_errs[d][δ][k]);
        M /= 2;
        for (int n = 0; n < M; n ++) c[n] = .5 * (c[2*n] + c[2*n + 1]);
        c.resize(M);
      }
    }
  }
}

void MetropolisCorrelation(FieldSquare& φ, const double& ε, const int& N, const int& disc, vector<vector<double>>& c_means, vector<vector<vector<double>>>& c_mses, vector<vector<vector<double>>>& c_mse_errs)
{
  auto dist = uniform_real_distribution<double>{-ε, ε};
  auto stand = uniform_real_distribution<double>{0., 1.};
  for (int n = 0; n < disc; n ++)
  {
    for (int y = 0; y < φ.L(); y ++)
    {
      for (int x = 0; x < φ.L(); x ++)
      {
        double curr = φ(x, y);
        double prop = curr + dist(r);
        double κ2γ = φ.kappa() * φ.kappa() * φ.gamma(x, y);
        double diff = pow(curr - κ2γ, 2.) - pow(prop - κ2γ, 2.);
        if (diff >= 0. || stand(r) < exp(diff / (2.*φ.kappa()*φ.kappa()))) φ(x, y) = prop;
      }
    }
    cout << n << "\n";
  }

  vector<vector<double>> Φs(2);
  for (int d = 0; d < 2; d ++)
  {
    Φs[d] = vector<double>(φ.L());
    for (int δ = 0; δ <= φ.L()/2; δ ++) c_means[d][δ] = 0.;
  }
  vector<vector<vector<double>>> cs(N);

  for (int n = 0; n < N; n ++)
  {
    for (int d = 0; d < 2; d ++)
    {
      for (int x = 0; x < φ.L(); x ++) Φs[d][x] = 0.;
    }

    for (int y = 0; y < φ.L(); y ++)
    {
      for (int x = 0; x < φ.L(); x ++)
      {
        double curr = φ(x, y);
        double prop = curr + dist(r);
        double κ2γ = φ.kappa() * φ.kappa() * φ.gamma(x, y);
        double diff = pow(curr - κ2γ, 2.) - pow(prop - κ2γ, 2.);
        if (diff >= 0. || stand(r) < exp(diff / (2.*φ.kappa()*φ.kappa())))
        {
          φ(x, y) = prop;
          Φs[0][x] += prop;
          Φs[1][y] += prop;
        }
        else
        {
          Φs[0][x] += curr;
          Φs[1][y] += curr;
        }
      }
    }

    cs[n] = vector<vector<double>>(2);
    for (int d = 0; d < 2; d ++)
    {
      cs[n][d] = vector<double>(φ.L()/2 + 1);
      for (int δ = 0; δ <= φ.L()/2; δ ++)
      {
        cs[n][d][δ] = 0.;
        for (int x = 0; x < φ.L(); x ++) cs[n][d][δ] += Φs[d][x] * Φs[d][(x+δ) % φ.L()];
        c_means[d][δ] += cs[n][d][δ];
      }
    }
    cout << n << "\n";
  }

  int K = int(.5 + log2(double(N)));
  for (int d = 0; d < 2; d ++)
  {
    for (int δ = 0; δ <= φ.L()/2; δ ++)
    {
      c_means[d][δ] /= double(N);

      vector<double> c(N);
      for (int n = 0; n < N; n ++) c[n] = cs[n][d][δ];
      int M = N;
      for (int k = 0; k < K; k ++)
      {
        mse_err(c, c_means[d][δ], c_mses[d][δ][k], c_mse_errs[d][δ][k]);
        M /= 2;
        for (int n = 0; n < M; n ++) c[n] = .5 * (c[2*n] + c[2*n + 1]);
        c.resize(M);
      }
    }
  }
}

void MetropolisCorrelation(FieldCube& φ, const double& ε, const int& N, const int& disc, vector<vector<double>>& c_means, vector<vector<vector<double>>>& c_mses, vector<vector<vector<double>>>& c_mse_errs)
{
  auto dist = uniform_real_distribution<double>{-ε, ε};
  auto stand = uniform_real_distribution<double>{0., 1.};
  for (int n = 0; n < disc; n ++)
  {
    for (int z = 0; z < φ.L(); z ++)
    {
      for (int y = 0; y < φ.L(); y ++)
      {
        for (int x = 0; x < φ.L(); x ++)
        {
          double curr = φ(x, y, z);
          double prop = curr + dist(r);
          double κ2γ = φ.kappa() * φ.kappa() * φ.gamma(x, y, z);
          double diff = pow(curr - κ2γ, 2.) - pow(prop - κ2γ, 2.);
          if (diff >= 0. || stand(r) < exp(diff / (2.*φ.kappa()*φ.kappa()))) φ(x, y, z) = prop;
        }
      }
    }
    cout << n << "\n";
  }

  vector<vector<double>> Φs(3);
  for (int d = 0; d < 3; d ++)
  {
    Φs[d] = vector<double>(φ.L());
    for (int δ = 0; δ <= φ.L()/2; δ ++) c_means[d][δ] = 0.;
  }
  vector<vector<vector<double>>> cs(N);

  for (int n = 0; n < N; n ++)
  {
    for (int d = 0; d < 3; d ++)
    {
      for (int x = 0; x < φ.L(); x ++) Φs[d][x] = 0.;
    }

    for (int z = 0; z < φ.L(); z ++)
    {
      for (int y = 0; y < φ.L(); y ++)
      {
        for (int x = 0; x < φ.L(); x ++)
        {
          double curr = φ(x, y, z);
          double prop = curr + dist(r);
          double κ2γ = φ.kappa() * φ.kappa() * φ.gamma(x, y, z);
          double diff = pow(curr - κ2γ, 2.) - pow(prop - κ2γ, 2.);
          if (diff >= 0. || stand(r) < exp(diff / (2.*φ.kappa()*φ.kappa())))
          {
            φ(x, y, z) = prop;
            Φs[0][x] += prop;
            Φs[1][y] += prop;
            Φs[2][z] += prop;
          }
          else
          {
            Φs[0][x] += curr;
            Φs[1][y] += curr;
            Φs[2][z] += curr;
          }
        }
      }
    }

    cs[n] = vector<vector<double>>(3);
    for (int d = 0; d < 3; d ++)
    {
      cs[n][d] = vector<double>(φ.L()/2 + 1);
      for (int δ = 0; δ <= φ.L()/2; δ ++)
      {
        cs[n][d][δ] = 0.;
        for (int x = 0; x < φ.L(); x ++) cs[n][d][δ] += Φs[d][x] * Φs[d][(x+δ) % φ.L()];
        c_means[d][δ] += cs[n][d][δ];
      }
    }
    cout << n << "\n";
  }

  int K = int(.5 + log2(double(N)));
  for (int d = 0; d < 3; d ++)
  {
    for (int δ = 0; δ <= φ.L()/2; δ ++)
    {
      c_means[d][δ] /= double(N);

      vector<double> c(N);
      for (int n = 0; n < N; n ++) c[n] = cs[n][d][δ];
      int M = N;
      for (int k = 0; k < K; k ++)
      {
        mse_err(c, c_means[d][δ], c_mses[d][δ][k], c_mse_errs[d][δ][k]);
        M /= 2;
        for (int n = 0; n < M; n ++) c[n] = .5 * (c[2*n] + c[2*n + 1]);
        c.resize(M);
      }
    }
  }
}

void Correlation(Field& φ, const double& ε, const int& N, const int& disc, vector<vector<double>>& c_means, vector<vector<vector<double>>>& c_mses, vector<vector<vector<double>>>& c_mse_errs)
{
  if (ε < 0) GibbsCorrelation(φ, N, disc, c_means, c_mses, c_mse_errs);
  else MetropolisCorrelation(φ, ε, N, disc, c_means, c_mses, c_mse_errs);
}

void Correlation(FieldSquare& φ, const double& ε, const int& N, const int& disc, vector<vector<double>>& c_means, vector<vector<vector<double>>>& c_mses, vector<vector<vector<double>>>& c_mse_errs)
{
  if (ε < 0) GibbsCorrelation(φ, N, disc, c_means, c_mses, c_mse_errs);
  else MetropolisCorrelation(φ, ε, N, disc, c_means, c_mses, c_mse_errs);
}

void Correlation(FieldCube& φ, const double& ε, const int& N, const int& disc, vector<vector<double>>& c_means, vector<vector<vector<double>>>& c_mses, vector<vector<vector<double>>>& c_mse_errs)
{
  if (ε < 0) GibbsCorrelation(φ, N, disc, c_means, c_mses, c_mse_errs);
  else MetropolisCorrelation(φ, ε, N, disc, c_means, c_mses, c_mse_errs);
}

int main()
{
  int K = 15;
  int N = int(.5 + pow(2, K));
  int L = 32;
  vector<int> Ls = { L, L };
  double am = .1;

  vector<vector<double>> c_means(2);
  vector<vector<vector<double>>> c_mses(2);
  vector<vector<vector<double>>> c_mse_errs(2);
  for (int d = 0; d < 2; d ++)
  {
    c_means[d] = vector<double>(L/2 + 1);
    c_mses[d] = vector<vector<double>>(L/2 + 1);
    c_mse_errs[d] = vector<vector<double>>(L/2 + 1);
    for (int δ = 0; δ <= L/2; δ ++)
    {
      c_mses[d][δ] = vector<double>(K);
      c_mse_errs[d][δ] = vector<double>(K);
    }
  }

  FieldSquare φ(L, am);
  GibbsCorrelation(φ, N, int(.1*N), c_means, c_mses, c_mse_errs);

  ofstream file1, file2;
  file1.open("d = 1.csv");
  file2.open("d = 2.csv");
  for (int δ = 0; δ <= L/2; δ ++)
  {
    file1 << c_means[0][δ] << ",";
    file2 << c_means[1][δ] << ",";
  }
  file1 << c_means[0][L/2];
  file2 << c_means[0][L/2];
  file1.close();
  file2.close();

  return 0;
}