#include "func.hpp"
{init_layers}
void set_values(){{
    {set_values}
}}
template <typename T, int inputRank, int outputRank>
inline Eigen::Tensor<T, outputRank> call_model(const Eigen::Tensor<T, inputRank> &input)
{{{body}
    return output;
}}
    