#include "func.hpp"
Eigen::array<int, 4> F_L{{1, 0, 2, 3}};
Eigen::array<int, 3> S_1D{1, 0, 2};
{init_layers}
void set_values(){{
    {set_values}
}}
template <typename T, int inputRank, int outputRank>
inline Eigen::Tensor<T, outputRank> call_model(const Eigen::Tensor<T, inputRank> &input)
{{{body}
    return output;
}}
    