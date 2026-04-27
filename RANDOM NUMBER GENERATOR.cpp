#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>

using namespace std;

int main() 
    {
    srand(time(0));
    double randomDecimal = 0.0 + static_cast<double>(rand()) / RAND_MAX;
    cout << fixed << setprecision(2) << randomDecimal << endl;
    return 0;
    }
