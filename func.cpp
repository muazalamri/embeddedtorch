#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <limits>
#include <array>
#include <vector>
#include <cassert>
using namespace Eigen;

// tensor adding
template <typename T, int Rank>
inline Eigen::Tensor<T, Rank> addTensors(const Eigen::Tensor<T, Rank> &A,
                                  const Eigen::Tensor<T, Rank> &B)
{
    assert(A.dimensions() == B.dimensions() && "Tensors must have same shape");
    return A + B;
}
// tensor reshaping
template <typename T, int Rank, int NewRank>
inline Eigen::Tensor<T, NewRank> reshapeTensor(const Eigen::Tensor<T, Rank> &A,
                                        const Eigen::array<Eigen::Index, NewRank> &newDims)
{
    return A.reshape(newDims);
}
// matrix * scalar
template <typename T, int Rank>
inline Eigen::Tensor<T, Rank> scaleTensor(const Eigen::Tensor<T, Rank> &A, T scalar)
{
    return A * scalar;
}
// tensor slicing
template <typename T, int Rank>
inline Eigen::Tensor<T, Rank> sliceTensor(const Eigen::Tensor<T, Rank> &A,
                                   const Eigen::array<Eigen::Index, Rank> &offsets,
                                   const Eigen::array<Eigen::Index, Rank> &extents)
{
    return A.slice(offsets, extents);
}
// tensor contraction (generalized matrix multiplication)
template <typename T, int RankA, int RankB, int RankC>
inline Eigen::Tensor<T, RankC> contractTensors(const Eigen::Tensor<T, RankA> &A,
                                        const Eigen::Tensor<T, RankB> &B,
                                        const Eigen::array<Eigen::IndexPair<int>, 1> &contractDims)
{
    return A.contract(B, contractDims);
}
// ReLU activation
template <typename T, int Rank>
inline Eigen::Tensor<T, Rank> relu(const Eigen::Tensor<T, Rank> &A)
{
    return A.cwiseMax(static_cast<T>(0));
}
// Sigmoid activation
template <typename T, int Rank>
inline Eigen::Tensor<T, Rank> sigmoid(const Eigen::Tensor<T, Rank> &A)
{
    return static_cast<T>(1) / (static_cast<T>(1) + (-A).exp());
}
// Tanh activation
template <typename T, int Rank>
inline Eigen::Tensor<T, Rank> tanh(const Eigen::Tensor<T, Rank> &A)
{
    return A.tanh();
}
// Softmax activation along specified axis//using average as a prototype
template <typename T, int Rank>
Eigen::Tensor<T, Rank> softmax(const Eigen::Tensor<T, Rank> &input, int axis)
{
    static_assert(Rank >= 1, "softmax: Rank must be >= 1");
    if (axis < 0 || axis >= Rank)
        throw std::invalid_argument("softmax: axis out of range");

    Eigen::array<int, 1> reduce_axis = {axis};

    // |x|
    auto abs_input = input.abs().eval();

    // mean(|x|) along axis
    auto sum_abs = abs_input.sum(reduce_axis).eval();
    T axis_size = static_cast<T>(input.dimension(axis));
    auto mean_abs = (sum_abs / sum_abs.constant(axis_size)).eval();

    // reshape reduced result to restore the axis (size 1)
    Eigen::array<Eigen::Index, Rank> reshape_dims;
    for (int i = 0; i < Rank; ++i)
        reshape_dims[i] = (i == axis) ? 1 : input.dimension(i);
    auto mean_reshaped = mean_abs.reshape(reshape_dims).eval();

    // broadcast mean back to full shape
    Eigen::array<int, Rank> bcast;
    for (int i = 0; i < Rank; ++i)
        bcast[i] = (i == axis) ? input.dimension(i) : 1;
    auto mean_bcast = mean_reshaped.broadcast(bcast).eval();

    // normalized |x| / mean(|x|)
    auto normalized = (abs_input / (mean_bcast + mean_bcast.constant(T(1e-12)))).eval();

    return normalized;
}
// Linear layer: output = input * weights^T + bias
template <typename T, int InputRank, int OutputRank>
Eigen::Tensor<T, OutputRank> linearLayer(const Eigen::Tensor<T, InputRank> &input,
                                         const Eigen::Tensor<T, 2> &weights,
                                         const Eigen::Tensor<T, 1> &bias)
{
    Eigen::array<int, OutputRank>
        bcast;
    bcast[0] = input.dimension(0);
    bcast[1] = 1;
    std::cout << "bias" << bias.dimensions() << std::endl;
    array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};
    std::cout << "mlut : " << input.dimensions() << "*" << weights.dimensions() << std::endl;
    Tensor<T, 2> output = contractTensors<T, 2, 2, 2>(input, weights, contract_dims);
    std::cout << "1111111111111" << std::endl;
    Eigen::array<Eigen::Index, 2> bias_dims = {1, bias.dimensions()[0]}; // bias.dimensions()[0])};
    Eigen::Tensor<T, 2> Td_bias = reshapeTensor<T, 1, 2>(bias, bias_dims).broadcast(bcast);
    std::cout << "dims : " << Td_bias.dimensions() << std::endl;
    return addTensors<float, 2>(output, Td_bias); // 2d_bias
}

// Max pooling (2D)
template <typename T, int InputRank, int OutputRank>
Eigen::Tensor<T, OutputRank> maxPool2D(const Eigen::Tensor<T, InputRank> &input,
                                       const Eigen::array<int, 2> &poolSize,
                                       const Eigen::array<int, 2> &strides)
{
    Eigen::array<int, InputRank> kernel;
    Eigen::array<int, InputRank> stride;
    for (int i = 0; i < InputRank - 2; ++i)
    {
        kernel[i] = 1;
        stride[i] = 1;
    }
    kernel[InputRank - 2] = poolSize[0];
    kernel[InputRank - 1] = poolSize[1];
    stride[InputRank - 2] = strides[0];
    stride[InputRank - 1] = strides[1];
    return input.extract_image_patches(poolSize[0], poolSize[1], strides[0], strides[1], 1, 1, 0).maximum(kernel).stride(stride);
}
// Average pooling (2D)
template <typename T, int InputRank, int OutputRank>
Eigen::Tensor<T, OutputRank> avgPool2D(const Eigen::Tensor<T, InputRank> &input,
                                       const Eigen::array<int, 2> &poolSize,
                                       const Eigen::array<int, 2> &strides)
{
    Eigen::array<int, InputRank> kernel;
    Eigen::array<int, InputRank> stride;
    for (int i = 0; i < InputRank - 2; ++i)
    {
        kernel[i] = 1;
        stride[i] = 1;
    }
    kernel[InputRank - 2] = poolSize[0];
    kernel[InputRank - 1] = poolSize[1];
    stride[InputRank - 2] = strides[0];
    stride[InputRank - 1] = strides[1];
    return input.extract_image_patches(poolSize[0], poolSize[1], strides[0], strides[1], 1, 1, 0).mean(kernel).stride(stride);
}
// Batch normalization
template <typename T, int Rank>
Eigen::Tensor<T, Rank> batchNorm(const Eigen::Tensor<T, Rank> &input,
                                 const Eigen::Tensor<T, 1> &mean,
                                 const Eigen::Tensor<T, 1> &variance,
                                 const Eigen::Tensor<T, 1> &gamma,
                                 const Eigen::Tensor<T, 1> &beta,
                                 T epsilon = static_cast<T>(1e-5))
{
    Eigen::array<int, Rank> bcast;
    for (int i = 0; i < Rank - 1; ++i)
        bcast[i] = input.dimensions()[i];
    bcast[Rank - 1] = 1;
    Eigen::Tensor<T, Rank> normalized = (input - mean.reshape(Eigen::array<Eigen::Index, Rank>{1}).broadcast(bcast)) /
                                        (variance.reshape(Eigen::array<Eigen::Index, Rank>{1}).broadcast(bcast) + epsilon).sqrt();
    return gamma.reshape(Eigen::array<Eigen::Index, Rank>{1}).broadcast(bcast) * normalized +
           beta.reshape(Eigen::array<Eigen::Index, Rank>{1}).broadcast(bcast);
}
// Dropout
template <typename T, int Rank>
Eigen::Tensor<T, Rank> dropout(const Eigen::Tensor<T, Rank> &input, T dropProb)
{
    assert(dropProb >= 0 && dropProb < 1 && "dropProb must be in [0, 1)");
    Eigen::Tensor<T, Rank> mask = (Eigen::Tensor<T, Rank>::Random(input.dimensions()) + static_cast<T>(1)) / static_cast<T>(2);
    mask = (mask > dropProb).template cast<T>();
    return input * mask / (static_cast<T>(1) - dropProb);
}
// Padding
template <typename T, int InputRank, int OutputRank>
Eigen::Tensor<T, OutputRank> padTensor(const Eigen::Tensor<T, InputRank> &input,
                                       const Eigen::array<std::pair<int, int>, InputRank> &padding)
{
    return input.pad(padding);
}

template <typename T, int InputRank, int OutputRank, int KernelRank, int StrideRank, int FilterRank>
Eigen::Tensor<T, OutputRank>
Conv(const Eigen::Tensor<T, InputRank> &input,
     const Eigen::Tensor<T, KernelRank> &kernel,
     const Eigen::array<int, StrideRank> &strides,
     const Eigen::array<std::pair<int, int>, InputRank> &padding)
{
    // Derive spatial rank from kernel rank:
    constexpr int SpatialRank = KernelRank - 2;
    static_assert(InputRank == SpatialRank + 1, "InputRank must be SpatialRank+1 (channels + spatial dims).");
    static_assert(OutputRank == SpatialRank + 1, "OutputRank must be SpatialRank+1 (filters + spatial dims).");
    static_assert(StrideRank == SpatialRank, "StrideRank must equal the number of spatial dimensions (SpatialRank).");

    // Helper index-unpack utilities for calling tensor(...)
    auto tensor_get = []<int Rank>(const Eigen::Tensor<T, Rank> &t, const std::array<int, Rank> &idx) -> T
    {
        return [&t, &idx]<std::size_t... I>(std::index_sequence<I...>) -> T
        {
            return t(idx[I]...);
        }(std::make_index_sequence<Rank>{});
    };
    // capture val as well
    auto tensor_set = []<int Rank>(Eigen::Tensor<T, Rank> &t,
                                   const std::array<int, Rank> &idx,
                                   const T &val)
    {
        // include &val in the capture list
        [&t, &idx, &val]<std::size_t... I>(std::index_sequence<I...>)
        {
            t(idx[I]...) = val;
        }(std::make_index_sequence<Rank>{});
    };

    // Read channels and dims
    const int C_in = static_cast<int>(input.dimension(0));
    const int C_out = static_cast<int>(kernel.dimension(0));
    const int kernel_C_in = static_cast<int>(kernel.dimension(1));
    assert(kernel_C_in == C_in && "kernel second dimension must match input channels");

    // Gather spatial dims
    std::array<int, SpatialRank> in_spatial_dims{};
    std::array<int, SpatialRank> kernel_spatial_dims{};
    for (int s = 0; s < SpatialRank; ++s)
    {
        in_spatial_dims[s] = static_cast<int>(input.dimension(s + 1));
        kernel_spatial_dims[s] = static_cast<int>(kernel.dimension(s + 2));
    }

    // Extract pad pairs and stride vector (strides length == SpatialRank)
    std::array<int, SpatialRank> pad_before{};
    std::array<int, SpatialRank> pad_after{};
    for (int s = 0; s < SpatialRank; ++s)
    {
        // padding[0] corresponds to channel axis and must be zero.
        if (s == -1)
        {
        } // no-op to keep pattern clear
        pad_before[s] = padding[s + 1].first;
        pad_after[s] = padding[s + 1].second;
    }

    // Compute output spatial dimensions:
    std::array<int, SpatialRank> out_spatial_dims{};
    for (int s = 0; s < SpatialRank; ++s)
    {
        int numerator = in_spatial_dims[s] + pad_before[s] + pad_after[s] - kernel_spatial_dims[s];
        if (numerator < 0)
            numerator = 0; // avoid negative
        out_spatial_dims[s] = numerator / strides[s] + 1;
    }

    // Build output DSizes and allocate output tensor
    Eigen::DSizes<Eigen::Index, OutputRank> out_sizes;
    out_sizes[0] = C_out;
    for (int s = 0; s < SpatialRank; ++s)
        out_sizes[s + 1] = out_spatial_dims[s];
    Eigen::Tensor<T, OutputRank> output(out_sizes);

    // Initialize output to zero
    output.setZero();

    // Multi-index helpers for iterating spatial positions
    // We'll iterate: for each out_channel, for each out_spatial_index (S dims), compute convolution sum.
    std::array<int, SpatialRank> out_idx{};
    std::array<int, SpatialRank> k_idx{};
    std::array<int, SpatialRank> in_pos{}; // computed input position for each spatial axis

    // Iterate output channels
    for (int oc = 0; oc < C_out; ++oc)
    {
        // Iterate all output spatial positions (multi-dimensional loop)
        // We'll implement a simple odometer-style iterator.
        std::array<int, SpatialRank> it{};
        for (int i = 0; i < SpatialRank; ++i)
            it[i] = 0;

        bool done = (SpatialRank == 0); // if zero spatial dims, single position
        while (!done)
        {
            // compute sum for this output coordinate (oc, it...)
            T accum = T(0);

            // loop over input channels
            for (int ic = 0; ic < C_in; ++ic)
            {
                // iterate kernel spatial positions
                std::array<int, SpatialRank> kit{};
                for (int i = 0; i < SpatialRank; ++i)
                    kit[i] = 0;
                bool kdone = (SpatialRank == 0);
                while (!kdone)
                {
                    // compute input position per spatial dim:
                    bool in_bounds = true;
                    for (int s = 0; s < SpatialRank; ++s)
                    {
                        int out_coord = it[s];
                        int stride = strides[s];
                        int pos = out_coord * stride - pad_before[s] + kit[s];
                        in_pos[s] = pos;
                        if (pos < 0 || pos >= in_spatial_dims[s])
                        {
                            in_bounds = false;
                            break;
                        }
                    }

                    if (in_bounds)
                    {
                        // build index arrays to access input and kernel
                        // input idx: [ic, in_pos...]
                        std::array<int, InputRank> in_index{};
                        in_index[0] = ic;
                        for (int s = 0; s < SpatialRank; ++s)
                            in_index[s + 1] = in_pos[s];

                        // kernel idx: [oc, ic, kit...]
                        std::array<int, KernelRank> k_index{};
                        k_index[0] = oc;
                        k_index[1] = ic;
                        for (int s = 0; s < SpatialRank; ++s)
                            k_index[s + 2] = kit[s];

                        // fetch values
                        T in_val = tensor_get.template operator()<InputRank>(input, in_index);
                        T k_val = tensor_get.template operator()<KernelRank>(kernel, k_index);

                        accum += in_val * k_val;
                    }

                    // increment k-index (kernel spatial odometer)
                    if (SpatialRank == 0)
                    {
                        kdone = true;
                    }
                    else
                    {
                        int dimi = 0;
                        while (dimi < SpatialRank)
                        {
                            ++kit[dimi];
                            if (kit[dimi] < kernel_spatial_dims[dimi])
                                break;
                            kit[dimi] = 0;
                            ++dimi;
                        }
                        if (dimi == SpatialRank)
                            kdone = true;
                    }
                } // kernel positions
            } // input channels

            // set output at (oc, it...)
            std::array<int, OutputRank> out_index{};
            out_index[0] = oc;
            for (int s = 0; s < SpatialRank; ++s)
                out_index[s + 1] = it[s];
            tensor_set.template operator()<OutputRank>(output, out_index, accum);

            // increment output spatial odometer
            if (SpatialRank == 0)
            {
                done = true;
            }
            else
            {
                int d = 0;
                while (d < SpatialRank)
                {
                    ++it[d];
                    if (it[d] < out_spatial_dims[d])
                        break;
                    it[d] = 0;
                    ++d;
                }
                if (d == SpatialRank)
                    done = true;
            }
        } // output spatial positions
    } // output channels

    return output;
}
#ifdef TEST_FUNC
#define TEST_FUNC_MAIN
int main()
{
    std::cout << "Eigen Tensor Operations Example" << std::endl;
    Tensor<float, 2> A(2, 3);
    A.setRandom();
    Tensor<float, 2> B(2, 3);
    B.setRandom();
    Tensor<float, 2> C = addTensors(A, B);
    std::cout << "A + B = \n"
              << C << std::endl;
    // reshape example
    Eigen::array<Eigen::Index, 2> newDims = {3, 2};
    Tensor<float, 2> reshapedA = reshapeTensor<float, 2, 2>(A, newDims);
    std::cout << "Reshaped A = \n"
              << reshapedA << std::endl;
    // scale example
    float scalar = 2.0f;
    Tensor<float, 2> scaledA = scaleTensor<float, 2>(A, scalar);
    std::cout << "Scaled A = \n"
              << scaledA << std::endl;
    // slice example
    Eigen::array<Eigen::Index, 2> offsets = {0, 1};
    Eigen::array<Eigen::Index, 2> extents = {2, 2};
    Tensor<float, 2> slicedA = sliceTensor<float, 2>(A, offsets, extents);
    std::cout << "Sliced A = \n"
              << slicedA << std::endl;
    // activation examples
    Tensor<float, 2> reluA = relu<float, 2>(A);
    std::cout << "ReLU(A) = \n"
              << reluA << std::endl;
    Tensor<float, 2> sigmoidA = sigmoid<float, 2>(A);
    std::cout << "Sigmoid(A) = \n"
              << sigmoidA << std::endl;
    Tensor<float, 2> tanhA = tanh<float, 2>(A);
    std::cout << "Tanh(A) = \n"
              << tanhA << std::endl;
    Tensor<float, 2> softmaxA = softmax<float, 2>(A, 1);
    std::cout << "Softmax(A) = \n"
              << softmaxA << std::endl;
    // linear layer example
    Tensor<float, 2> weights(3, 4);
    weights.setRandom();
    Tensor<float, 1> bias(4);
    bias.setRandom();
    Tensor<float, 2> linearOut = linearLayer<float, 2, 2>(A, weights, bias);
    std::cout << "Linear Layer Output = \n"
              << linearOut << std::endl;
    // contractTensors example
    Tensor<float, 2> D(3, 4);
    D.setRandom();
    Tensor<float, 2> E = contractTensors<float, 2, 2, 2>(A, D, {IndexPair<int>(1, 0)});
    std::cout << "A * D = \n"
              << E << std::endl;
    return 0;
}
#endif
#endif