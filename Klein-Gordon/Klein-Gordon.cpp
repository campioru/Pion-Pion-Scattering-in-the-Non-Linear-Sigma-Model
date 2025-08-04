#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
using namespace std;

mt19937 r(1234);
uniform_real_distribution<double> standard_uniform(0., 1.);
normal_distribution<double> standard_normal(0., 1.);

class Field
{
  private:
    int d_, L_total_;
    vector<int> L_;
    double m_;
    vector<double> φ_;
    vector<vector<int>> nei_;

    int index_(const vector<int>& coord) const
    {
      int index = coord.front(), Π = L_.front();
      for (int d = 1; d < d_; d ++)
      {
        index += coord[d] * Π;
        Π *= L_[d];
      }
      return index;
    }

    double gamma_(const int& x) const
    {
      double γ = 0.;
      for (int d = 0; d < d_; d ++) γ += φ_[nei_[x][d]] + φ_[nei_[x][d_+d]];
      return γ;
    }
    double gamma_pos_(const int& x) const
    {
      double γ = 0.;
      for (int d = 0; d < d_; d ++) γ += φ_[nei_[x][d]];
      return γ;
    }

  public:
    Field(const vector<int>& L, const double& m) : d_(L.size()), L_total_(1), L_(d_), m_(m), φ_(0), nei_(0)
    {
      L_ = L;
      for (int d = 0; d < d_; d ++) L_total_ *= L_[d];
      φ_.resize(L_total_, 0.);

      nei_.resize(L_total_, vector<int>(2*d_));
      vector<int> coord(d_, 0);
      for (int x = 0; x < L_total_; x ++)
      {
        for (int d = 0; d < d_; d ++)
        {
          coord[d] = (coord[d]+1) % L_[d];
          nei_[x][d] = index_(coord);
          coord[d] = (coord[d]-2+L_[d]) % L_[d];
          nei_[x][d_+d] = index_(coord);
          coord[d] = (coord[d]+1) % L_[d];
        }

        coord.front() += 1;
        for (int d = 0; d < d_-1; d ++)
        {
          if (coord[d] == L_[d])
          {
            coord[d] = 0;
            coord[d+1] += 1;
          }
          else break;
        }
      }
    }

    double& operator() (const int& x)       { return φ_[x]; }
    double  operator() (const int& x) const { return φ_[x]; }
    double& m()       { return m_; }
    double  m() const { return m_; }
    int d() const { return d_; }
    int L_total() const { return L_total_; }
    vector<int> L()             const { return L_;    }
    int         L(const int& d) const { return L_[d]; }
    double kappa() const { return pow(2.*d_ + m_*m_, -.5); }

    double act() const
    {
      double S = 0., fac = d_ + m_*m_/2.;
      for (int x = 0; x < L_total_; x ++) S += φ_[x] * (fac*φ_[x] - gamma_pos_(x));
      return S;
    }
    vector<double> Phi() const
    {
      int L_x_ = L_total_ / L_.back();
      vector<double> Φ(L_.back(), 0.);
      for (int t = 0; t < L_.back(); t ++)
      {
        for (int x = 0; x < L_x_; x ++) Φ[t] += φ_[t*L_x_ + x];
      }
      return Φ;
    }

    double MA_Sweep(double& S, const double& ε)
    {
      double α = 0;
      double κ2 = 1. / (2.*d_ + m_*m_);
      for (int x = 0; x < L_total_; x ++)
      {
        double curr = φ_[x];
        double prop = curr + 2*ε*standard_uniform(r) - ε;
        double κ2γ = κ2*gamma_(x);
        double ΔS = (pow(prop - κ2γ, 2.) - pow(curr - κ2γ, 2.))/(2*κ2);
        if (ΔS <= 0. || standard_uniform(r) < exp(-ΔS))
        {
          φ_[x] = prop;
          S += ΔS;
          α += 1;
        }
      }
      return α/L_total_;
    }
    void GS_Sweep(double& S)
    {
      double κ = kappa();
      double κ2 = κ*κ;
      for (int x = 0; x < L_total_; x ++)
      {
        double κ2γ = κ2*gamma_(x);
        double prop = κ*standard_normal(r) + κ2γ;
        double curr = φ_[x];
        φ_[x] = prop;
        S += (pow(prop - κ2γ, 2.) - pow(curr - κ2γ, 2.))/(2*κ2);
      }
    }
};


vector<double> corr(const vector<double>& Φ)
{
  vector<double> c(Φ.size()/2 + 1, 0.);
  for (int δ = 0; δ <= Φ.size()/2; δ ++)
  {
    for (int t = 0; t < Φ.size(); t ++) c[δ] += Φ[t]*Φ[(t+δ) % Φ.size()];
  }
  return c;
}
double mag(const vector<double>& Φ)
{
  double M = 0.;
  for (int t = 0; t < Φ.size(); t ++) M += Φ[t];
  return M;
}

double mean(const vector<double>& chain)
{
  double sum = 0.;
  for (int n = 0; n < chain.size(); n ++) sum += chain[n];
  return sum/chain.size();
}
double quad_sum(const vector<double>& chain, const double& μ)
{
  double sum = 0.;
  for (int n = 0; n < chain.size(); n ++) sum += pow(chain[n] - μ, 2.);
  return sum;
}
double mean_error(const vector<double>& chain, const double& μ)
{
  return sqrt(quad_sum(chain, μ) / (chain.size()*(chain.size()-1.)));
}

void autocorr(const vector<double>& chain, const double& μ, double& τ, double& Δτ)
{
  int N = chain.size(), M = 0;
  double fac = 2.*N/quad_sum(chain, μ);
  τ = 1.;
  do
  {
    M++;
    double sum = 0.;
    for (int n = 0; n < N-M; n ++) sum += (chain[n]-μ)*(chain[n+M]-μ);
    τ += sum/(N-M)*fac;
  } while (M < 5*τ);
  Δτ = τ*sqrt((4.*M+2.)/N);
}
