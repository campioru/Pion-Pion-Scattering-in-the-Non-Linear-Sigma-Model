#include <iostream>
#include <fstream>
#include <vector>
#include <valarray>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
using namespace std;

thread_local mt19937 r;
thread_local uniform_real_distribution<double> standard_uniform(0., 1.);
thread_local normal_distribution<double> standard_normal(0., 1.);
thread_local exponential_distribution<double> standard_exponential(1.);


class SU2
{
  private:
    valarray<double> u_;
  public:
    SU2(const valarray<double>& u) : u_(u) { }
    double operator[] (const int& i) const { return u_[i]; }
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
    int d_, L_total_;
    vector<int> L_;
    double β_, λ_;
    vector<SU2> φ_;
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

    valarray<double> Sigma_(const int& x) const
    {
      valarray<double> Σ = { λ_, 0., 0., 0. };
      for (int d = 0; d < d_; d ++) Σ += φ_[nei_[x][d]].comp() + φ_[nei_[x][d_+d]].comp(); // for (int y : nei_[x]) Σ += φ_[y].comp();
      return Σ;
    }
    valarray<double> Sigma_pos_(const int& x) const
    {
      valarray<double> Σ = { λ_, 0., 0., 0. };
      for (int d = 0; d < d_; d ++) Σ += φ_[nei_[x][d]].comp();
      return Σ;
    }

    void pos_evol_(const vector<valarray<double>>& π_, const double& ε)
    {
      #pragma omp parallel for
      for (int x = 0; x < L_total_; x ++)
      {
        valarray<double> α = ε*π_[x];
        double mag = sqrt((α*α).sum());
        double sinα = sin(mag)/mag;
        φ_[x] = SU2({cos(mag), sinα*α[0], sinα*α[1], sinα*α[2]}) * φ_[x];
      }
    }
    void mom_evol_(vector<valarray<double>>& π_, const double& ε)
    {
      #pragma omp parallel for
      for (int x = 0; x < L_total_; x ++)
      {
        valarray<double> u = φ_[x].comp(), Σ = Sigma_(x);
        π_[x] -= β_*ε*valarray<double>{
          -u[0]*Σ[1] + u[1]*Σ[0] + u[2]*Σ[3] - u[3]*Σ[2],
          -u[0]*Σ[2] - u[1]*Σ[3] + u[2]*Σ[0] + u[3]*Σ[1],
          -u[0]*Σ[3] + u[1]*Σ[2] - u[2]*Σ[1] + u[3]*Σ[0],
        };
      }
    }

  public:
    Field(const vector<int>& L, const double& β, const double& λ) : d_(L.size()), L_total_(1), L_(L), β_(β), λ_(λ)
    {
      for (int d = 0; d < d_; d ++) L_total_ *= L_[d]; // for (int l : L_) L_total_ *= l;
      φ_.resize(L_total_, SU2({ 1., 0., 0., 0. }));

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

    SU2& operator() (const int& x)       { return φ_[x]; }
    SU2  operator() (const int& x) const { return φ_[x]; }
    double& beta()       { return β_; }
    double  beta() const { return β_; }
    double& lambda()       { return λ_; }
    double  lambda() const { return λ_; }
    int d() const { return d_; }
    int L_total() const { return L_total_; }
    vector<int> L()             const { return L_;    }
    int         L(const int& d) const { return L_[d]; }

    double act() const
    {
      double sum = 0.;
      #pragma omp parallel for reduction(+:sum)
      for (int x = 0; x < L_total_; x ++) sum += (Sigma_pos_(x)*φ_[x].comp()).sum();
      return -β_*sum;
    }
    vector<valarray<double>> Phi() const
    {
      int L_x_ = L_total_ / L_.back();
      vector<valarray<double>> Φ(L_.back(), valarray<double>(0., 4));
      #pragma omp parallel for
      for (int t = 0; t < L_.back(); t ++)
      {
        for (int x = 0; x < L_x_; x ++) Φ[t] += φ_[t*L_x_ + x].comp();
      }
      return Φ;
    }

    int GS_Sweep(double& S)
    {
      int gen = 0;
      #pragma omp parallel reduction(+:S,gen)
      {
        for (int c = 0; c < 2; c ++) // for (vector<int> board : checker_)
        {
          #pragma omp for
          for (int y = 0; y < checker_[c].size(); y ++) // for (int x : board)
          {
            int x = checker_[c][y];

            valarray<double> Σ = Sigma_(x);
            double sqrtdet = sqrt((Σ*Σ).sum());

            valarray<double> v(4);
            double U;
            double ρ = β_*sqrtdet;
            do
            {
              double A = standard_normal(r) / sqrt(2.);
              U = standard_uniform(r);
              v[0] = 1. - (A*A + standard_exponential(r)) / ρ;
              gen += 1;
            } while (2*U*U > v[0] + 1.);

            v[1] = 2.*standard_uniform(r) - 1.;
            double θ = 2.*M_PI*standard_uniform(r);
            double fac = sqrt(1. - v[1]*v[1]);
            v[2] = cos(θ)*fac;
            v[3] = sin(θ)*fac;

            fac = sqrt(1. - v[0]*v[0]);
            for (int i = 1; i < 4; i ++) v[i] *= fac;

            valarray<double> curr = φ_[x].comp();
            φ_[x] = SU2(v) * SU2(Σ / sqrtdet);
            S += β_*((Σ*(curr - φ_[x].comp())).sum());
          }
        }
      }
      return gen;
    }

    int MA_Sweep(double& S)
    {
      int α = 0;
      #pragma omp parallel reduction(+:S,α)
      {
        for (int c = 0; c < 2; c ++) // for (vector<int> board : checker_)
        {
          #pragma omp for
          for (int y = 0; y < checker_[c].size(); y ++) // for (int x : board)
          {
            int x = checker_[c][y];

            valarray<double> prop(4);
            for (int i = 0; i < 4; i ++) prop[i] = standard_normal(r); // for (double& prop_i : prop)
            prop /= sqrt((prop*prop).sum());
            double ΔS = β_*((Sigma_(x)*(φ_[x].comp() - prop)).sum());
            if (ΔS <= 0. || standard_uniform(r) < exp(-ΔS))
            {
              φ_[x] = SU2(prop);
              S += ΔS;
              α += 1;
            }
          }
        }
      }
      return α;
    }

    int HMC_Sweep(double& S, const double& Δt, const int& Nt)
    {
      vector<valarray<double>> π_(L_total_, valarray<double>(3));
      #pragma omp parallel for collapse(2)
      for (int x = 0; x < L_total_; x ++) for (int i = 0; i < 3; i ++) π_[x][i] = standard_normal(r); // for (valarray<double>& π_x : π_) for (double& π_x_i : π_x)
      vector<valarray<double>> π = π_;
      vector<SU2> φ = φ_;
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
      for (int x = 0; x < L_total_; x ++) ΔH += (π_[x]*π_[x] - π[x]*π[x]).sum();
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


vector<valarray<double>> corr(const vector<valarray<double>>& Φ)
{
  vector<valarray<double>> c(Φ.size()/2 + 1, valarray<double>(0., 4));
  #pragma omp parallel for
  for (int δ = 0; δ <= Φ.size()/2; δ ++)
  {
    for (int t = 0; t < Φ.size(); t ++) c[δ] += Φ[t]*Φ[(t+δ) % Φ.size()];
  }
  return c;
}
valarray<double> mag(const vector<valarray<double>>& Φ)
{
  valarray<double> M(0., 4);
  // #pragma omp parallel for reduction(+:M)
  for (int t = 0; t < Φ.size(); t ++) M += Φ[t]; // for (double Φ_t : Φ)
  return M;
}

double mean(const vector<double>& chain)
{
  double sum = 0.;
  #pragma omp parallel for reduction(+:sum)
  for (int n = 0; n < chain.size(); n ++) sum += chain[n]; // for (double chain_n : chain)
  return sum/chain.size();
}
double quad_sum(const vector<double>& chain, const double& μ)
{
  double sum = 0.;
  #pragma omp parallel for reduction(+:sum)
  for (int n = 0; n < chain.size(); n ++) sum += pow(chain[n] - μ, 2.); // for (double chain_n : chain)
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
    #pragma omp parallel for reduction(+:sum)
    for (int n = 0; n < N-M; n ++) sum += (chain[n]-μ)*(chain[n+M]-μ);
    τ += sum/(N-M)*fac;
  } while (M < 5*τ);
  Δτ = τ*sqrt((4.*M+2.)/N);
}
