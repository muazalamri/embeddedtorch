#include "func.cpp"
#include <unsupported/Eigen/CXX11/Tensor>
template <typename T, int in_features, int out_features>
struct LinearLayer
{
    Eigen::Tensor<T, 2> weight; // shape: (out_features, in_features)
    Eigen::Tensor<T, 1> bias;   // shape: (out_features)

    LinearLayer(const Eigen::Tensor<T, 2> &weight_, const Eigen::Tensor<T, 1> &bias_)
        : weight(weight_), bias(bias_) {}
    Eigen::Tensor<T, 2> call(const Eigen::Tensor<T, 2> &input)
    {
        return linearLayer<T, 2, 2>(input, weight, bias);
    };
};