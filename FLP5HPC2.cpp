#include <iostream>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <iomanip>
using namespace std;
using namespace chrono;

void print_array(int arr[], int size)
{
    cout << "Array: [ ";
    for (int i = 0; i < size; i++) 
    {
        cout << arr[i] << " ";
    }
    cout << "]\n";
}

void sequential_bubble_sort(int arr[], int size, double &seq_time) 
{
    int temp[size];
    copy(arr, arr + size, temp);

    auto start = high_resolution_clock::now();
    for (int i = 0; i < size - 1; i++) 
    {
        for (int j = 0; j < size - i - 1; j++) 
        {
            if (temp[j] > temp[j + 1]) 
            {
                swap(temp[j], temp[j + 1]);
            }
        }
    }
    auto end = high_resolution_clock::now();
    seq_time = duration<double>(end - start).count();
    cout << "\nSequential Bubble Sort Time: " << fixed << setprecision(10) << seq_time << " s\n";
    print_array(temp, size);
}

void parallel_odd_even_sort(int arr[], int size, double &seq_time) 
{
    int temp[size];
    copy(arr, arr + size, temp);

    auto start = high_resolution_clock::now();

    bool sorted = false;
    while (!sorted) 
    {
        sorted = true;

        #pragma omp parallel for shared(temp, size, sorted)
        for (int i = 1; i < size - 1; i += 2) 
        {
            if (temp[i] > temp[i + 1]) 
            {
                swap(temp[i], temp[i + 1]);
                sorted = false;
            }
        }

        #pragma omp parallel for shared(temp, size, sorted)
        for (int i = 0; i < size - 1; i += 2) 
        {
            if (temp[i] > temp[i + 1]) 
            {
                swap(temp[i], temp[i + 1]);
                sorted = false;
            }
        }
    }

    auto end = high_resolution_clock::now();
    double par_time = duration<double>(end - start).count();
    cout << endl;
    cout << "Parallel Bubble Sort Time: " 
         << fixed << setprecision(10) << par_time << " s\n";
    print_array(temp, size);
    cout << endl;
    cout << "Speedup: " << fixed << setprecision(10) << seq_time / par_time << "\n";
}

void merge(int arr[], int left, int mid, int right) 
{
    int temp[right - left + 1], i = left, j = mid + 1, k = 0;
    while (i <= mid && j <= right) 
    {
        temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];
    }
    while (i <= mid) 
    {
        temp[k++] = arr[i++];
    }
    while (j <= right) 
    {
        temp[k++] = arr[j++];
    }
    copy(temp, temp + k, arr + left);
}

void sequential_merge_sort(int arr[], int left, int right) 
{
    if (left < right) 
    {
        int mid = (left + right) / 2;
        sequential_merge_sort(arr, left, mid);
        sequential_merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void parallel_merge_sort(int arr[], int left, int right) 
{
    if (left < right) 
    {
        int mid = (left + right) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            parallel_merge_sort(arr, left, mid);
            #pragma omp section
            parallel_merge_sort(arr, mid + 1, right);
        }
        merge(arr, left, mid, right);
    }
}

void perform_merge_sort(int arr[], int size, bool parallel, double &seq_time) 
{
    int temp[size];
    copy(arr, arr + size, temp);

    auto start = high_resolution_clock::now();
    if (parallel)
    {
        parallel_merge_sort(temp, 0, size - 1);
    }
    else
    {
        sequential_merge_sort(temp, 0, size - 1);
    }
    auto end = high_resolution_clock::now();

    double time_taken = duration<double>(end - start).count();
    cout << endl;
    cout << (parallel ? "Parallel" : "Sequential") << " Merge Sort Time: " << fixed << setprecision(10) << time_taken << " s\n";
    print_array(temp, size);
    cout<<endl;
    if (parallel)
    {
        cout << "Speedup: " << fixed << setprecision(10) << seq_time / time_taken << "\n";
    }
    else
    {
        seq_time = time_taken;
    }
}

int main() 
{
    int size;
    int choice;

    cout << "Enter size of array: ";
    cin >> size;

    int arr[size];
    for (int &x : arr) x = rand() % size;

    double seq_bubble_time, seq_merge_time;

    while (true) 
    {
        cout<<"\n-------------------------";
        cout << "\n         Menu\n";
        cout<<"-------------------------\n";
        cout << " 1. Bubble Sort\n";
        cout << " 2. Merge Sort\n";
        cout << " 3. Exit\n";
        cout<<"-------------------------\n";
        cout << " Enter your choice: ";
        cin >> choice;
        cout<<endl;
        switch (choice) 
        {
        case 1:
            cout << "\nInput Array: ";
            print_array(arr, size);
            sequential_bubble_sort(arr, size, seq_bubble_time);
            parallel_odd_even_sort(arr, size, seq_bubble_time);
            break;
        case 2:
            cout << "\nInput Array: ";
            print_array(arr, size);
            perform_merge_sort(arr, size, false, seq_merge_time);
            perform_merge_sort(arr, size, true, seq_merge_time);
            break;
        case 3:
            cout << "Exiting Program\n";
            return 0;
        default: 
            cout << "Invalid choice! Please try again.\n";
        }
    }
}

