#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
using namespace std;

auto r = mt19937 {123};

void mse_err(const vector<double>& vec, const double& mean, double& mse, double& err);
double normal_sample(const double& μ, const double& σ);

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

void GibbsCorrelation(FieldSquare& φ, const int& N, vector<vector<double>>& c_means)
{
  vector<vector<double>> Φs(2);
  for (int d = 0; d < 2; d ++)
  {
    Φs[d] = vector<double>(φ.L());
    for (int δ = 0; δ <= φ.L()/2; δ ++) c_means[d][δ] = 0.;
  }

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

    for (int d = 0; d < 2; d ++)
    {
      for (int δ = 0; δ <= φ.L()/2; δ ++)
      {
        for (int x = 0; x < φ.L(); x ++) c_means[d][δ] += Φs[d][x] * Φs[d][(x+δ) % φ.L()];
      }
    }
  }

  for (int d = 0; d < 2; d ++)
  {
    for (int δ = 0; δ <= φ.L()/2; δ ++) c_means[d][δ] /= double(N);
  }
}

void GibbsCorrelation(FieldCube& φ, const int& N, vector<vector<double>>& c_means)
{
  vector<vector<double>> Φs(3);
  for (int d = 0; d < 3; d ++)
  {
    Φs[d] = vector<double>(φ.L());
    for (int δ = 0; δ <= φ.L()/2; δ ++) c_means[d][δ] = 0.;
  }

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

    for (int d = 0; d < 3; d ++)
    {
      for (int δ = 0; δ <= φ.L()/2; δ ++)
      {
        for (int x = 0; x < φ.L(); x ++) c_means[d][δ] += Φs[d][x] * Φs[d][(x+δ) % φ.L()];
      }
    }
  }

  for (int d = 0; d < 3; d ++)
  {
    for (int δ = 0; δ <= φ.L()/2; δ ++) c_means[d][δ] /= double(N);
  }
}

void MetropolisCorrelation(FieldSquare& φ, const double& ε, const int& N, vector<vector<double>>& c_means)
{
  auto dist = uniform_real_distribution<double>{-ε, ε};
  auto stand = uniform_real_distribution<double>{0., 1.};

  vector<vector<double>> Φs(2);
  for (int d = 0; d < 2; d ++)
  {
    Φs[d] = vector<double>(φ.L());
    for (int δ = 0; δ <= φ.L()/2; δ ++) c_means[d][δ] = 0.;
  }

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

    for (int d = 0; d < 2; d ++)
    {
      for (int δ = 0; δ <= φ.L()/2; δ ++)
      {
        for (int x = 0; x < φ.L(); x ++) c_means[d][δ] += Φs[d][x] * Φs[d][(x+δ) % φ.L()];
      }
    }
  }

  for (int d = 0; d < 2; d ++)
  {
    for (int δ = 0; δ <= φ.L()/2; δ ++) c_means[d][δ] /= double(N);
  }
}

void MetropolisCorrelation(FieldCube& φ, const double& ε, const int& N, vector<vector<double>>& c_means)
{
  auto dist = uniform_real_distribution<double>{-ε, ε};
  auto stand = uniform_real_distribution<double>{0., 1.};

  vector<vector<double>> Φs(3);
  for (int d = 0; d < 3; d ++)
  {
    Φs[d] = vector<double>(φ.L());
    for (int δ = 0; δ <= φ.L()/2; δ ++) c_means[d][δ] = 0.;
  }

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

    for (int d = 0; d < 3; d ++)
    {
      for (int δ = 0; δ <= φ.L()/2; δ ++)
      {
        for (int x = 0; x < φ.L(); x ++) c_means[d][δ] += Φs[d][x] * Φs[d][(x+δ) % φ.L()];
      }
    }
  }

  for (int d = 0; d < 3; d ++)
  {
    for (int δ = 0; δ <= φ.L()/2; δ ++) c_means[d][δ] /= double(N);
  }
}
