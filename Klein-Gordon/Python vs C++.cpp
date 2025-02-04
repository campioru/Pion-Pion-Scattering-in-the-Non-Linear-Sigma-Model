#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
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

vector<int> coord_init(const int& length)
{
  vector<int> coord(length);
  for (int d = 0; d < length; d ++) coord[d] = 0;
  return coord;
}

void coord_update(vector<int>& coord, Field& φ)
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


void GibbsCorrelation(Field& φ, const int& N, vector<vector<double>>& c_means)
{
  vector<vector<double>> Φs(φ.d());
  for (int d = 0; d < φ.d(); d ++)
  {
    Φs[d] = vector<double>(φ.L(d));
    for (int δ = 0; δ <= φ.L(d)/2; δ ++) c_means[d][δ] = 0.;
  }

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

    for (int d = 0; d < φ.d(); d ++)
    {
      for (int δ = 0; δ <= φ.L(d)/2; δ ++)
      {
        for (int x = 0; x < φ.L(d); x ++) c_means[d][δ] += Φs[d][x] * Φs[d][(x+δ) % φ.L(d)];
      }
    }
  }

  for (int d = 0; d < φ.d(); d ++)
  {
    for (int δ = 0; δ <= φ.L(d)/2; δ ++) c_means[d][δ] /= double(N);
  }
}

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

void MetropolisCorrelation(Field& φ, const double& ε, const int& N, vector<vector<double>>& c_means)
{
  auto dist = uniform_real_distribution<double>{-ε, ε};
  auto stand = uniform_real_distribution<double>{0., 1.};

  vector<vector<double>> Φs(φ.d());
  for (int d = 0; d < φ.d(); d ++)
  {
    Φs[d] = vector<double>(φ.L(d));
    for (int δ = 0; δ <= φ.L(d)/2; δ ++) c_means[d][δ] = 0.;
  }

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

    for (int d = 0; d < φ.d(); d ++)
    {
      for (int δ = 0; δ <= φ.L(d)/2; δ ++)
      {
        for (int x = 0; x < φ.L(d); x ++) c_means[d][δ] += Φs[d][x] * Φs[d][(x+δ) % φ.L(d)];
      }
    }
  }

  for (int d = 0; d < φ.d(); d ++)
  {
    for (int δ = 0; δ <= φ.L(d)/2; δ ++) c_means[d][δ] /= double(N);
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

void Correlation(Field& φ, const double& ε, const int& N, vector<vector<double>>& c_means)
{
  if (ε < 0) GibbsCorrelation(φ, N, c_means);
  else MetropolisCorrelation(φ, ε, N, c_means);
}

void Correlation(FieldSquare& φ, const double& ε, const int& N, vector<vector<double>>& c_means)
{
  if (ε < 0) GibbsCorrelation(φ, N, c_means);
  else MetropolisCorrelation(φ, ε, N, c_means);
}

void Correlation(FieldCube& φ, const double& ε, const int& N, vector<vector<double>>& c_means)
{
  if (ε < 0) GibbsCorrelation(φ, N, c_means);
  else MetropolisCorrelation(φ, ε, N, c_means);
}

int main()
{
  int L = 10;
  double am = .1;
  vector<double> εs = { -1., 1. };
  int K = 18;
  vector<int> Ns(K);
  for (int k = 0; k < K; k ++) Ns[k] = int(.5 + pow(2., k));
  chrono::duration<double> times[2][2][2][K];

  vector<vector<vector<vector<vector<double>>>>> c_means(2);
  for (int d = 0; d < 2; d ++)
  {
    c_means[d] = vector<vector<vector<vector<double>>>>(2);
    for (int o = 0; o < 2; o ++)
    {
      c_means[d][o] = vector<vector<vector<double>>>(2);
      for (int ε = 0; ε < 2; ε ++)
      {
        c_means[d][o][ε] = vector<vector<double>>(d+2);
        for (int j = 0; j < d+2; j ++)
        {
          c_means[d][o][ε][j] = vector<double>(L/2 + 1);
        }
      }
    }
  }

  for (int ε = 0; ε < 2; ε ++)
  {
    for (int k = 0; k < K; k ++)
    {
      {auto start = chrono::steady_clock::now();
      Field phi({L, L}, am);
      Correlation(phi, εs[ε], Ns[k], c_means[0][0][ε]);
      times[0][0][ε][k] = chrono::steady_clock::now() - start;
      cout << "s g " << εs[ε] << " " << Ns[k] << "\n";}

      {auto start = chrono::steady_clock::now();
      FieldSquare phi(L, am);
      Correlation(phi, εs[ε], Ns[k], c_means[0][1][ε]);
      times[0][1][ε][k] = chrono::steady_clock::now() - start;
      cout << "s o " << εs[ε] << " " << Ns[k] << "\n";}

      {auto start = chrono::steady_clock::now();
      Field phi({L, L, L}, am);
      Correlation(phi, εs[ε], Ns[k], c_means[1][0][ε]);
      times[1][0][ε][k] = chrono::steady_clock::now() - start;
      cout << "c g " << εs[ε] << " " << Ns[k] << "\n";}

      {auto start = chrono::steady_clock::now();
      FieldCube phi(L, am);
      Correlation(phi, εs[ε], Ns[k], c_means[1][1][ε]);
      times[1][1][ε][k] = chrono::steady_clock::now() - start;
      cout << "c o " << εs[ε] << " " << Ns[k] << "\n";}
    }
  }

  ofstream mean_file, time_file;
  mean_file.open("means.txt");
  time_file.open("times.txt");
  mean_file << "Square/cube, general/optimised, Gibbs/Metropolis-Hastings, dimension: data(separation)\nN = " << Ns[K-1] << "\n\n";
  time_file << "Square/cube, general/optimised, Gibbs/Metropolis-Hastings: data(iterations)\nN = " << Ns[K-1] << "\n\n";

  for (int d = 0; d < 2; d ++)
  {
    for (int o = 0; o < 2; o ++)
    {
      for (int ε = 0; ε < 2; ε ++)
      {
        for (int j = 0; j < d+2; j ++)
        {
          mean_file << d << ", " << o << ", " << ε << ", " << j+1 << ": ";
          for (int δ = 0; δ <= L/2; δ ++)
          {
            mean_file << c_means[d][o][ε][j][δ] << ",";
          }
          mean_file << "\n";
        }

        time_file << d << ", " << o << ", " << ε << ": ";
        for (int k = 0; k < K; k ++)
        {
          time_file << times[d][o][ε][k].count() << ",";
        }
        time_file << "\n";
      }
    }
  }

  mean_file.close();
  time_file.close();

  return 0;
}