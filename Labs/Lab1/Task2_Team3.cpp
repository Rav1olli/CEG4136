
#include <CL/sycl.hpp>
#include <math.h>
#include <algorithm>

using namespace sycl;

// Activation function
float sigmoid(float x) {
    float res;
    res = 1 / (1 + exp(x * -1));
    return res;
}

int main() {

    queue q{ cpu_selector() };

    // Declaring the matrix sizes
    constexpr size_t rowZ = 1;
    constexpr size_t colZ = 3;

    constexpr size_t rowW = 3;
    constexpr size_t colW = 2;

    constexpr size_t rowB = 1;
    constexpr size_t colB = 2;

    constexpr size_t rowA = 1;
    constexpr size_t colA = 2;

    //Declaring the matrices
    std::vector<float> Z(rowZ * colZ);
    std::vector<float> W(rowW * colW);
    std::vector<float> B(rowB * colB);
    std::vector<float> A(rowA * colA);

    //Declaring and initializing the matrix data in arrays
    //PLEASE NOTE: The values used for the input matrix were taken from the output of TASK 2
    //Here we must put the A matrix from Task 2 through an activation function (sigmoid)
    std::array<float, 3> arrZ = { sigmoid(0.7), sigmoid(0.7), sigmoid(1.1) };
    std::array<float, 6> arrW = { 0.1, 0.4, 0.2, 0.5, 0.3, 0.6 };
    std::array<float, 2> arrB = { 0.1, 0.2 };

    //Filling each vector with the appropriate data using a lambda function
    int i = 0;
    auto dataZ = [arrZ, &i]() {
        return arrZ[i++];
    };
    std::generate(Z.begin(), Z.end(), dataZ);

    i = 0;
    auto dataW = [arrW, &i]() {
        return arrW[i++];
    };
    std::generate(W.begin(), W.end(), dataW);

    i = 0;
    auto dataB = [arrB, &i]() {
        return arrB[i++];
    };
    std::generate(B.begin(), B.end(), dataB);

    std::fill(A.begin(), A.end(), 0.0);

    {
        //Declaring the 2D buffers for each matrix
        buffer<float, 2> Z_buf(Z.data(), range<2>(rowZ, colZ));
        buffer<float, 2> W_buf(W.data(), range<2>(rowW, colW));
        buffer<float, 2> B_buf(B.data(), range<2>(rowB, colB));
        buffer<float, 2> A_buf(A.data(), range<2>(rowA, colA));

        //Submit the code to be run in parallel
        q.submit([&](handler& h) {
            //Declare the accessors for the buffers
            accessor Z{ Z_buf, h };
            accessor W{ W_buf, h };
            accessor B{ B_buf, h };
            accessor A{ A_buf, h };

            //Calculating the resulting matrix
            h.parallel_for(range{ rowZ, colW }, [=](id<2> idx) {
                int j = idx[0];
                int i = idx[1];

                for (int k = 0; k < colZ; ++k) {
                    A[j][i] += Z[j][k] * W[k][i];
                }

                A[j][i] += B[j][i];

                });
            });
    }

    //Print out the result
    for (auto i : A)
        std::cout << i << "  ";
    std::cout << "\nProgram ends\n";

    return 0;
}