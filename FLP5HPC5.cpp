#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <omp.h>

using namespace std;

void sequential_lr(const vector<double>& x, const vector<double>& y, double& beta0, double& beta1, double& time_taken) 
    {
    int n = x.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < n; ++i) 
        {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        }
    beta1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    beta0 = (sum_y - beta1 * sum_x) / n;
    auto end = chrono::high_resolution_clock::now();

    time_taken = chrono::duration<double>(end - start).count();
    }

void parallel_lr(const vector<double>& x, const vector<double>& y, double& beta0, double& beta1, double& time_taken) 
    {
    int n = x.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;

    auto start = chrono::high_resolution_clock::now();
#pragma omp parallel for reduction(+ : sum_x, sum_y, sum_xy, sum_x2)
    for (int i = 0; i < n; ++i) 
        {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        }
    beta1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    beta0 = (sum_y - beta1 * sum_x) / n;
    auto end = chrono::high_resolution_clock::now();

    time_taken = chrono::duration<double>(end - start).count();
}

int main() 
    {
    vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    vector<double> y = {2.0, 4.0, 5.0, 4.0, 5.0};

    double beta0_seq, beta1_seq, time_seq;
    double beta0_par, beta1_par, time_par;

    // Sequential Linear Regression
    sequential_lr(x, y, beta0_seq, beta1_seq, time_seq);

    // Parallel Linear Regression
    parallel_lr(x, y, beta0_par, beta1_par, time_par);

    //cout << fixed << setprecision(2);
    cout << "Sequential Execution:\n";
    cout << "beta0: " << beta0_seq << ", beta1: " << beta1_seq << ", Time: " << time_seq << "s\n";
    cout << "Equation (Sequential): y = " << beta1_seq << "x + " << beta0_seq << "\n\n";

    cout << "Parallel Execution:\n";
    cout << "beta0: " << beta0_par << ", beta1: " << beta1_par << ", Time: " << time_par << "s\n";
    cout << "Equation (Parallel): y = " << beta1_par << "x + " << beta0_par << "\n\n";

    double speedup = (time_par > 0) ? (time_seq / time_par) : 0;
    cout << "Speedup: " << speedup << "\n";

    return 0;
    }

    
