#include <iostream>
#include <vector>

using namespace std;

// Function to print elements of a vector
void printData(const vector<int> &w){
    for(auto i : w)
        cout << i << " ";
    cout << endl;
}

int main() {
    // Create a vector of integers
    vector<int> v = {1, 2, 3, 4};  // or auto v = vector{1,2,3,4} in C++17

    // Print the vector
    printData(v);  // Output: 1 2 3 4

    return 0;
}
