#include<iostream>
#include<omp.h>
#include<iomanip>
#include<limits>
#include<time.h>
#include<chrono>
using namespace std;

const int n = 1000; // Size of the array

int sum_serial(int arr[], int n) 
    {
    int sum = 0;
    for (int i = 0; i <= n; ++i)
        {
        sum += arr[i];
        }
    return sum;
    }

int sum_parallel(int arr[], int n) 
    {
    int sum = 0;
    #pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i <= n; ++i)
        {
        sum += arr[i];
        }
    return sum;
    }

int average_serial(int arr[], int n) 
    {
    int sum = 0;
    for (int i = 0; i <= n; ++i)
        {
        sum += arr[i];
        }
    return sum / (n + 1);
    }

int average_parallel(int arr[], int n) 
    {
    int sum = 0;
    #pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i <= n; ++i)
        {
        sum += arr[i];
        }
    return sum / (n + 1);
    }

int min_serial(int arr[], int n) 
    {
    int min_val = arr[0];
    for (int i = 1; i <= n; ++i)
        {
        if (arr[i] < min_val)
            {
            min_val = arr[i];
            }
        }
    return min_val;
    }

int min_parallel(int arr[], int n) 
    {
    int min_val = arr[0];
    #pragma omp parallel for reduction(min : min_val)
    for (int i = 1; i <= n; ++i)
        {
        if (arr[i] < min_val)
            {
            min_val = arr[i];
            }
        }
    return min_val;
    }

int max_serial(int arr[], int n) 
    {
    int max_val = arr[0];
    for (int i = 1; i <= n; ++i)
        {
        if (arr[i] > max_val)
            {
            max_val = arr[i];
            }
        }
    return max_val;
    }

int max_parallel(int arr[], int n) 
    {
    int max_val = arr[0];
    #pragma omp parallel for reduction(max : max_val)
    for (int i = 1; i <= n; ++i)
        {
        if (arr[i] > max_val)
            {
            max_val = arr[i];
            }
        }
    return max_val;
    }

int main() 
    {
    int arr[n + 1];
    srand(time(0));
    for (int i = 0; i <= n; ++i)
        {
        arr[i] = rand() % 1000; // Random numbers between 0 and 999
        }
        
    // SUM
    auto ss_start = chrono::high_resolution_clock::now();
    int sum_s = sum_serial(arr, n);
    auto ss_end = chrono::high_resolution_clock::now();

    auto sp_start = chrono::high_resolution_clock::now();
    int sum_p = sum_parallel(arr, n);
    auto sp_end = chrono::high_resolution_clock::now();

    chrono::duration<double> d_ss = ss_end - ss_start;
    chrono::duration<double> d_sp = sp_end - sp_start;

    cout << "1. SUM\n\n";
    cout << "Serial result: " << sum_s << endl;
    cout << "Parallel result: " << sum_p << endl;
    cout << "Serial duration: " << fixed << setprecision(10) << d_ss.count() << " seconds\n";
    cout << "Parallel duration: " << fixed << setprecision(10) << d_sp.count() << " seconds\n";
    cout << "Speedup: " << d_ss.count() / d_sp.count() << "\n\n";

    // AVERAGE
    auto as_start = chrono::high_resolution_clock::now();
    int avg_s = average_serial(arr, n);
    auto as_end = chrono::high_resolution_clock::now();

    auto ap_start = chrono::high_resolution_clock::now();
    int avg_p = average_parallel(arr, n);
    auto ap_end = chrono::high_resolution_clock::now();

    chrono::duration<double> d_as = as_end - as_start;
    chrono::duration<double> d_ap = ap_end - ap_start;

    cout << "2. AVERAGE\n\n";
    cout << "Serial result: " << avg_s << endl;
    cout << "Parallel result: " << avg_p << endl;
    cout << "Serial duration: " << d_as.count() << " seconds\n";
    cout << "Parallel duration: " << d_ap.count() << " seconds\n";
    cout << "Speedup: " << d_as.count() / d_ap.count() << "\n\n";

    // MIN
    auto mins_start = chrono::high_resolution_clock::now();
    int min_s = min_serial(arr, n);
    auto mins_end = chrono::high_resolution_clock::now();

    auto minp_start = chrono::high_resolution_clock::now();
    int min_p = min_parallel(arr, n);
    auto minp_end = chrono::high_resolution_clock::now();

    chrono::duration<double> d_mins = mins_end - mins_start;
    chrono::duration<double> d_minp = minp_end - minp_start;

    cout << "3. MIN\n\n";
    cout << "Serial result: " << min_s << endl;
    cout << "Parallel result: " << min_p << endl;
    cout << "Serial duration: " << d_mins.count() << " seconds\n";
    cout << "Parallel duration: " << d_minp.count() << " seconds\n";
    cout << "Speedup: " << d_mins.count() / d_minp.count() << "\n\n";

    // MAX
    auto maxs_start = chrono::high_resolution_clock::now();
    int max_s = max_serial(arr, n);
    auto maxs_end = chrono::high_resolution_clock::now();

    auto maxp_start = chrono::high_resolution_clock::now();
    int max_p = max_parallel(arr, n);
    auto maxp_end = chrono::high_resolution_clock::now();

    chrono::duration<double> d_maxs = maxs_end - maxs_start;
    chrono::duration<double> d_maxp = maxp_end - maxp_start;

    cout << "4. MAX\n\n";
    cout << "Serial result: " << max_s << endl;
    cout << "Parallel result: " << max_p << endl;
    cout << "Serial duration: " << d_maxs.count() << " seconds\n";
    cout << "Parallel duration: " << d_maxp.count() << " seconds\n";
    cout << "Speedup: " << d_maxs.count() / d_maxp.count() << "\n\n";

    return 0;
}
