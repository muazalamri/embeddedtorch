#include "func.cpp"
#include <unsupported/Eigen/CXX11/Tensor>
template <typename T, int in_features, int out_features>
struct LinearLayer
{
    Eigen::Tensor<T, 2> weight; // shape: (out_features, in_features)
    Eigen::Tensor<T, 1> bias;   // shape: (out_features)

    LinearLayer(const Eigen::Tensor<T, 2> &weight_, const Eigen::Tensor<T, 1> &bias_)
        : weight(weight_), bias(bias_) {}
    Eigen::Tensor<T, 2> call(const Eigen::Tensor<T, 2> &input) {
        linearLayer<T, 2, 2>(input, weight, bias);
    };
};
void model(){
    //exmple
    Eigen::Tensor<float, 4, 3> weight({-0.2407853603363037, -0.21487325429916382, -0.006769835948944092, -0.3393939733505249, -0.03775608539581299, 0.4500913619995117, -0.09379267692565918, 0.09332132339477539, -0.12061834335327148, 0.22719407081604004, -0.3635868430137634, 0.2775450348854065});
    Eigen::Tensor<float, 3> bias({-0.32207608222961426, -0.41004037857055664, -0.21790432929992676});
}