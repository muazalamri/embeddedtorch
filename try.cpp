//#define DEBUG
#include <iostream>
#include "out/code.cpp"
#include <chrono>
int main()
{
    int batch=50;
    set_values();
    std::cout << "seted" << std::endl;
    Tensor<float, 2> input(batch,16);
    std::cout << "init" << std::endl;
    input.setRandom();
    std::cout << "rand" << std::endl;
    int retry = 1000000;
    std::cout << "starting" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < retry; i++)
    {
        call_model<float, 2, 2>(input);
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Total elapsed time in seconds (double)
    std::chrono::duration<double> elapsed = end - start;

    // Average time per iteration in milliseconds
    double avg_ms = (elapsed.count() * 1000.0) / (retry*batch);

    std::cout << "Average time over " << retry
              << " runs: " << avg_ms << " ms" << std::endl;
    std::cout << "done" << std::endl;
    return 0;
}