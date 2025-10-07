#include "func.cpp"
#include "classes.cpp"
{init_layers}
template <typename T, int inputRank, int outputRank>
Eigen::Tensor<T, inputRank> call_model(const Eigen::Tensor<T, inputRank> &input)
{{{body}
    return output;
}}
    