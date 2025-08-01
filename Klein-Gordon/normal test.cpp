#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <valarray>
#include <cmath>
#include <random>
#include <chrono>
using namespace std;

auto r = mt19937 {123};

void halfref(int& N)
{
    N /= 2;
    cout << "func " << N << "\n";
}

void half(int N)
{
    N /= 2;
    cout << "func " << N << "\n";
}

class SU2
{
  private:
    valarray<double> u_;
  public:
    SU2(const valarray<double>& u) : u_(u) {} // test if works
    double& operator[] (const int& i)       { return u_[i]; }
    double  operator[] (const int& i) const { return u_[i]; }
    SU2 operator* (const SU2& v) const { return SU2({
      u_[0]*v[0] - u_[1]*v[1] - u_[2]*v[2] - u_[3]*v[3],
      u_[0]*v[1] + u_[1]*v[0] - u_[2]*v[3] + u_[3]*v[2],
      u_[0]*v[2] + u_[1]*v[3] + u_[2]*v[0] - u_[3]*v[1],
      u_[0]*v[3] - u_[1]*v[2] + u_[2]*v[1] + u_[3]*v[0]}); }
    /* vector<vector<complex<double>>> matrix() const { return {
      { u_[0] + 1i*u_[3], u_[2] + 1i*u_[1] },
      { -u_[2] + 1i*u_[1], u_[0] - 1i*u_[3] } }; } */
};

int main()
{
    auto r = mt19937 {123};
    auto norm = normal_distribution<double>{0.,4.};
    /*double vals[100000];
    for (int i = 0; i < 100000; i ++) vals[i] = norm(r);
    double mean = 0.;
    for (int i = 0; i < 100000; i ++) mean += vals[i];
    mean /= 100000.;
    double var = 0.;
    for (int i = 0; i < 100000; i ++) var += pow(vals[i] - mean, 2.);
    var /= 99999.;
    cout << mean << " " << var;

    return 0; */

    /* auto uni = uniform_real_distribution<double>{0.,1.};
    double vals[100000];
    for (int i = 0; i < 100000; i ++) vals[i] = uni(r);
    double mean = 0.;
    for (int i = 0; i < 100000; i ++) mean += vals[i];
    mean /= 100000.;
    double var = 0.;
    for (int i = 0; i < 100000; i ++) var += pow(vals[i] - mean, 2.);
    var /= 99999.;
    cout << mean << " " << var; */

    /* vector<vector<double>> bleh;
    bleh.push_back(vector<double>(2));
    bleh.push_back(vector<double>(4));
    cout << bleh.size() << "\n";
    cout << bleh[0].size() << "\n";
    cout << bleh[1].size() << "\n"; */

    /* vector<vector<double>> bleh(2);
    bleh[0] = vector<double>(2);
    bleh[1] = vector<double>(4);
    cout << bleh.size() << "\n";
    cout << bleh[0].size() << "\n";
    cout << bleh[1].size() << "\n"; */

    /* int N = 4;

    half(N);
    cout << N << "\n";
    halfref(N);
    cout << N << "\n"; */

    /* int i = 10;

    for (i = 0; i < 50; i ++) cout << i << "\n";

    cout << "after " << i << "\n"; */

    /* double phis[10][10][10];
    cout << sizeof(phis)/sizeof(double) << "\n";
    cout << sizeof(phis[0])/sizeof(double) << "\n";
    cout << sizeof(phis[0][0])/sizeof(double) << "\n";
    cout << sizeof(phis[0][0][0])/sizeof(double) << "\n"; */

    /* int test1 = 0;
    float test2 = 0.;
    double test3 = 0.;
    string s1 = typeid(test1).name();
    string s2 = typeid(test2).name();
    string s3 = typeid(test3).name();
    if (s1 == "i") cout << 1 << "\n";
    if (s2 == "i") cout << 2 << "\n";
    if (s3 == "i") cout << 3 << "\n"; */

    /* double start, fin, dub;

    start = time(NULL);
    for (int i = 0; i < 250000000; i ++) dub = uniform_real_distribution<double>{0., 1.}(r);
    fin = time(NULL);
    cout << fin - start << "\n";

    start = time(NULL);
    auto stand = uniform_real_distribution{0., 1.};
    for (int i = 0; i < 250000000; i ++) dub = stand(r);
    fin = time(NULL);
    cout << fin - start << "\n"; */

    /* {auto start = chrono::steady_clock::now();
    double cs[10000][2][5];
    for (int i = 0; i < 10000; i ++)
    {
        for (int j = 0; j < 2; j ++)
        {
            for (int k = 0; k < 5; k ++)
            {
                cs[i][j][k] = 0.;
                for (int l = 0; l < 10; l ++) cs[i][j][k] += norm(r);
            }
        }
    }
    auto fin = chrono::steady_clock::now();
    chrono::duration<double> total = fin - start;
    cout << total.count() << "\n";}

    {auto start = chrono::steady_clock::now();
    vector<vector<vector<double>>> cs(10000);
    for (int i = 0; i < 10000; i ++)
    {
        cs[i] = vector<vector<double>>(2);
        for (int j = 0; j < 2; j ++)
        {
            cs[i][j] = vector<double>(5);
            for (int k = 0; k < 5; k ++)
            {
                cs[i][j][k] = 0.;
                for (int l = 0; l < 10; l ++) cs[i][j][k] += norm(r);
            }
        }
    }
    auto fin = chrono::steady_clock::now();
    chrono::duration<double> total = fin - start;
    cout << total.count() << "\n";} */

    /* {auto start = chrono::steady_clock::now();
    valarray<double> val1(1000000);
    valarray<double> val2(1000000);
    for (int i = 0; i < 100000; i ++)
    {
        val1[i] = 0.;
        val2[i] = double(i);
    }
    for (int i = 0; i < 1000; i ++)
    {
        val1 += val2;
    }
    auto fin = chrono::steady_clock::now();
    chrono::duration<double> total = fin - start;
    cout << "valarray sum time " << total.count() << "\n";}

    {auto start = chrono::steady_clock::now();
    vector<double> vec1(1000000);
    vector<double> vec2(1000000);
    for (int i = 0; i < 100000; i ++)
    {
        vec1[i] = 0.;
        vec2[i] = double(i);
    }
    for (int i = 0; i < 1000; i ++)
    {
        for (int j = 0; j < 1000000; j ++) vec1[i] += vec2[i];
    }
    auto fin = chrono::steady_clock::now();
    chrono::duration<double> total = fin - start;
    cout << "valarray sum time " << total.count() << "\n";} */

    SU2 test({1.,0.,0.,0.});

    test[0] += 7.;
    cout << test[0];

    return 0;
}