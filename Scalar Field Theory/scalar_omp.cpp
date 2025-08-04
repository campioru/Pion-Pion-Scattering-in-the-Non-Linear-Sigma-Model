#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <omp.h>
using namespace std;

thread_local mt19937 r;
thread_local uniform_real_distribution<double> standard_uniform(0., 1.);
thread_local normal_distribution<double> standard_normal(0., 1.);

class Field
{
  private:
    int d_, L_total_, L_x_;
    vector<int> L_;
    double m_, λ_, κ_, κ2_, κ2in_, fac_;
    vector<double> φ_;
    vector<vector<int>> nei_, checker_;

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

    void pos_evol_(const vector<double>& π_, const double& ε)
    {
      #pragma omp parallel for
      for (int x = 0; x < L_total_; x ++) φ_[x] += ε*π_[x];
    }
    void mom_evol_(vector<double>& π_, const double& ε) const
    {
      #pragma omp parallel for
      for (int x = 0; x < L_total_; x ++) π_[x] -= ε*(κ2in_*φ_[x] - gamma_(x) + 4.*λ_*pow(φ_[x], 3.));
    }

  public:
    Field(const vector<int>& L, const double& m, const double& λ) : d_(L.size()), L_total_(1), L_(d_), m_(m), λ_(λ), κ2in_(2.*L.size() + m*m), φ_(0), nei_(0), checker_(0)
    {
      L_ = L;
      for (int d = 0; d < d_; d ++) L_total_ *= L_[d];
      L_x_ = L_total_ / L_.back();
      κ_ = pow(κ2in_, -.5);
      κ2_ = 1/κ2in_;
      fac_ = κ2in_/2.;
      φ_.resize(L_total_, 0.);

      nei_.resize(L_total_, vector<int>(2*d_));
      checker_.resize(2);
      vector<int> coord(d_, 0);
      for (int x = 0; x < L_total_; x ++)
      {
        int coord_sum = 0;
        for (int d = 0; d < d_; d ++)
        {
          coord[d] = (coord[d]+1) % L_[d];
          nei_[x][d] = index_(coord);
          coord[d] = (coord[d]-2+L_[d]) % L_[d];
          nei_[x][d_+d] = index_(coord);
          coord[d] = (coord[d]+1) % L_[d];

          coord_sum += coord[d];
        }

        if (coord_sum % 2) checker_[1].push_back(x);
        else checker_[0].push_back(x);

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
    /* double& m()       { return m_; } */
    double  m() const { return m_; }
    void mass_update(const double& m)
    {
      m_ = m;
      κ2in_ = 2.*d_ + m_*m_;
      κ_ = pow(κ2in_, -.5);
      κ2_ = 1/κ2in_;
      fac_ = κ2in_/2.;
    }
    double& lambda()       { return λ_; }
    double  lambda() const { return λ_; } // maybe change name
    int d() const { return d_; }
    int L_total() const { return L_total_; }
    vector<int> L()             const { return L_;    }
    int         L(const int& d) const { return L_[d]; }
    double kappa() const { return κ_; }

    double act() const
    {
      double S = 0.;
      #pragma omp parallel for reduction(+:S)
      for (int x = 0; x < L_total_; x ++) S += φ_[x] * (fac_*φ_[x] - gamma_pos_(x)) + λ_*pow(φ_[x], 4.);
      return S;
    }
    vector<double> Phi() const
    {
      vector<double> Φ(L_.back(), 0.);
      #pragma omp parallel for
      for (int t = 0; t < L_.back(); t ++)
      {
        for (int x = 0; x < L_x_; x ++) Φ[t] += φ_[t*L_x_ + x];
      }
      return Φ;
    }

    int MA_Sweep(double& S, const double& ε)
    {
      int α = 0;
      #pragma omp parallel reduction(+:S,α)
      {
        double ΔS_thread = 0.;
        int α_thread = 0;
        for (int i = 0; i < 2; i ++)
        {
          #pragma omp for
          for (int x = 0; x < checker_[i].size(); x ++)
          {
            double curr = φ_[checker_[i][x]];
            double prop = curr + 2*ε*standard_uniform(r) - ε;
            double κ2γ = κ2_*gamma_(checker_[i][x]);
            double ΔS = (pow(prop - κ2γ, 2.) - pow(curr - κ2γ, 2.))/(2*κ2_) + λ_*(pow(prop, 4.) - pow(curr, 4.));
            if (ΔS <= 0. || standard_uniform(r) < exp(-ΔS))
            {
              φ_[checker_[i][x]] = prop;
              ΔS_thread += ΔS;
              α_thread += 1;
            }
          }
          if (i == 0)
          {
            #pragma omp barrier // maybe tidy via private checker function & call twice
          }
        }
        S += ΔS_thread;
        α += α_thread;
      }
      return α;
    }

    int GS_Sweep(double& S)
    {
      int gen = 0;
      #pragma omp parallel reduction(+:S,gen)
      {
        double ΔS_thread = 0.;
        int gen_thread = 0;
        for (int i = 0; i < 2; i ++)
        {
          #pragma omp for
          for (int x = 0; x < checker_[i].size(); x ++)
          {
            double prop, rej_term;
            double κ2γ = κ2_*gamma_(checker_[i][x]);
            do
            {
              prop = κ_*standard_normal(r) + κ2γ;
              gen_thread += 1;
              rej_term = λ_*pow(prop, 4.);
            } while (standard_uniform(r) > exp(-rej_term));
            double curr = φ_[checker_[i][x]];
            φ_[checker_[i][x]] = prop;
            ΔS_thread += (pow(prop - κ2γ, 2.) - pow(curr - κ2γ, 2.))/(2*κ2_) + rej_term - λ_*pow(curr, 4.);
          }
          if (i == 0)
          {
            #pragma omp barrier
          }
        }
        S += ΔS_thread;
        gen += gen_thread;
      }
      return gen;
    }

    int HMC_Sweep(double& S, const double& Δt, const int& Nt)
    {
      vector<double> π_(L_total_);
      #pragma omp parallel for
      for (int x = 0; x < L_total_; x ++) π_[x] = standard_normal(r);
      vector<double> π = π_, φ = φ_;
      pos_evol_(π_, Δt/2.);
      for (int nt = 1; nt < Nt; nt ++)
      {
        mom_evol_(π_, Δt);
        pos_evol_(π_, Δt);
      }
      mom_evol_(π_, Δt);
      pos_evol_(π_, Δt/2.);
      double ΔS = act() - S, ΔH = 0.;
      #pragma omp parallel for reduction(+:ΔH)
      for (int x = 0; x < L_total_; x ++) ΔH += π_[x]*π_[x] - π[x]*π[x];
      ΔH = .5*ΔH + ΔS;
      if (ΔH <= 0. || standard_uniform(r) < exp(-ΔH))
      {
        S += ΔS;
        return 1;
      }
      else
      {
        φ_ = φ;
        return 0;
      }
    }
};


vector<double> corr(const vector<double>& Φ)
{
  vector<double> c(Φ.size()/2 + 1, 0.);
  #pragma omp parallel for
  for (int δ = 0; δ <= Φ.size()/2; δ ++)
  {
    for (int t = 0; t < Φ.size(); t ++) c[δ] += Φ[t]*Φ[(t+δ) % Φ.size()];
  }
  return c;
}
double mag(const vector<double>& Φ)
{
  double M = 0.;
  #pragma omp parallel for reduction(+:M)
  for (int t = 0; t < Φ.size(); t ++) M += Φ[t];
  return M;
}

double mean(const vector<double>& chain)
{
  double sum = 0.;
  #pragma omp parallel for reduction(+:sum)
  for (int n = 0; n < chain.size(); n ++) sum += chain[n];
  return sum/chain.size();
}
double quad_sum(const vector<double>& chain, const double& μ)
{
  double sum = 0.;
  #pragma omp parallel for reduction(+:sum)
  for (int n = 0; n < chain.size(); n ++) sum += pow(chain[n] - μ, 2.);
  return sum;
}
double mean_error(const vector<double>& chain, const double& μ)
{
  return pow(quad_sum(chain, μ) / (chain.size()*(chain.size()-1.)), .5);
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
    #pragma omp parallel for reduction(+:sum)
    for (int n = 0; n < N-M; n ++) sum += (chain[n]-μ)*(chain[n+M]-μ);
    τ += sum/(N-M)*fac;
  } while (M < 5*τ);
  Δτ = τ*pow((4.*M+2.)/N, .5);
}
