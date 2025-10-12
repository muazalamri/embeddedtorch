#ifndef func_h
#define func_h
#ifdef DEBUG
#include <iostream>
#include <cassert>
#endif
#include <unsupported/Eigen/CXX11/Tensor>
#include <limits>
#include <array>
#include <vector>
using namespace Eigen;
// using threads
//  tensor adding
template <typename T, int Rank>
inline Eigen::Tensor<T, Rank> addTensors(const Eigen::Tensor<T, Rank> &A,
                                         const Eigen::Tensor<T, Rank> &B)
{
    #ifdef DEBUG
    std::cout << "addTensors input rank:" << Rank << std::endl;
    std::cout << "addTensors input dims:" << A.dimensions() << "+" << B.dimensions() << Rank << std::endl;
    assert(A.dimensions() == B.dimensions() && "Tensors must have same shape");
    #endif
    return A + B;
}
// tensor reshaping
template <typename T, int Rank, int NewRank>
inline Eigen::Tensor<T, NewRank> reshapeTensor(const Eigen::Tensor<T, Rank> &A,
                                               const Eigen::array<Eigen::Index, NewRank> &newDims)
{
    #ifdef DEBUG
    std::cout << "old dims: " << A.dimensions() << ", new dims: " << newDims;
    #endif
    return A.reshape(newDims);
}
// matrix * scalar
template <typename T, int Rank>
inline Eigen::Tensor<T, Rank> scaleTensor(const Eigen::Tensor<T, Rank> &A, T scalar)
{
    #ifdef DEBUG
    std::cout << "scaleTensor input rank: " << Rank << " ,input dims:" << A.dimensions() << std::endl;
    #endif
    return A * scalar;
}
// tensor slicing
template <typename T, int Rank>
inline Eigen::Tensor<T, Rank> sliceTensor(const Eigen::Tensor<T, Rank> &A,
                                          const Eigen::array<Eigen::Index, Rank> &offsets,
                                          const Eigen::array<Eigen::Index, Rank> &extents)
{
    #ifdef DEBUG
    std::cout << "sliceTensor input rank:" << Rank << ", input dims:" << A.dimensions() << std::endl;
    #endif
    return A.slice(offsets, extents);
}
// tensor contraction (generalized matrix multiplication)
template <typename T, int RankA, int RankB, int RankC>
inline Eigen::Tensor<T, RankC> contractTensors(const Eigen::Tensor<T, RankA> &A,
                                               const Eigen::Tensor<T, RankB> &B,
                                               const Eigen::array<Eigen::IndexPair<int>, 1> &contractDims)
{
    #ifdef DEBUG
    std::cout << "contractTensors input ranks: " << RankA << "," << RankB << " ,input dims:" << A.dimensions() << "*" << B.dimensions() << std::endl;
    #endif
    return A.contract(B, contractDims);
}
// ReLU activation
template <typename T, int Rank>
inline Eigen::Tensor<T, Rank> relu(const Eigen::Tensor<T, Rank> &A)
{
    #ifdef DEBUG
    std::cout << "relu input rank:" << Rank << " ,input dims:" << A.dimensions() << std::endl;
    #endif
    return A.cwiseMax(static_cast<T>(0));
}
// Sigmoid activation
template <typename T, int Rank>
inline Eigen::Tensor<T, Rank> sigmoid(const Eigen::Tensor<T, Rank> &A)
{
    #ifdef DEBUG
    std::cout << "sigmoid input rank:" << Rank << " ,input dims:" << A.dimensions() << std::endl;
    #endif
    return static_cast<T>(1) / (static_cast<T>(1) + (-A).exp());
}
// Tanh activation
template <typename T, int Rank>
inline Eigen::Tensor<T, Rank> tanh(const Eigen::Tensor<T, Rank> &A)
{
    #ifdef DEBUG
    std::cout << "tanh input rank:" << Rank << " ,input dims:" << A.dimensions() << std::endl;
    #endif
    return A.tanh();
}
// Softmax activation along specified axis//using average as a prototype
template <typename T, int Rank>
Eigen::Tensor<T, Rank> softmax(const Eigen::Tensor<T, Rank> &input, int axis)
{
    #ifdef DEBUG
    std::cout << "softmax input rank:" << Rank << std::endl;
    static_assert(Rank >= 1, "softmax: Rank must be >= 1");
    if (axis < 0 || axis >= Rank)
        throw std::invalid_argument("softmax: axis out of range");
    #endif
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
// Linear layer: output = input * weights + bias
template <typename T, int InputRank, int OutputRank>
Eigen::Tensor<T, OutputRank> linearLayer(const Eigen::Tensor<T, InputRank> &input,
                                         const Eigen::Tensor<T, 2> &weights,
                                         const Eigen::Tensor<T, 1> &bias)
{
    Eigen::array<int, OutputRank>
        bcast;
    bcast[0] = input.dimension(0);
    bcast[1] = 1;
    #ifdef DEBUG
    std::cout << "bias" << bias.dimensions() << std::endl;
    #endif
    array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};
    #ifdef DEBUG
    std::cout << "mlut : " << input.dimensions() << "*" << weights.dimensions() << std::endl;
    #endif
    Tensor<T, 2> output = contractTensors<T, 2, 2, 2>(input, weights, contract_dims);
    Eigen::array<Eigen::Index, 2> bias_dims = {1, bias.dimensions()[0]}; // bias.dimensions()[0])};
    Eigen::Tensor<T, 2> Td_bias = reshapeTensor<T, 1, 2>(bias, bias_dims).broadcast(bcast);
    #ifdef DEBUG
    std::cout << "bias dims : " << Td_bias.dimensions() << std::endl;
    #endif
    return addTensors<float, 2>(output, Td_bias); // 2d_bias
}

template <typename T>
Tensor<float, 4> maxPool2D(const Tensor<float, 4> &input,
                           int kernelH, int kernelW,
                           int strideH, int strideW,
                           bool same_padding = false)
{
    #ifdef DEBUG
    std::cout << "maxPool2D input dims:" << input.dimensions() << std::endl;
    #endif
    const int batch = input.dimension(0);
    const int inH = input.dimension(1);
    const int inW = input.dimension(2);
    const int channels = input.dimension(3);

    int outH, outW;
    int padTop = 0, padLeft = 0;

    if (same_padding)
    {
        outH = static_cast<int>(std::ceil(float(inH) / strideH));
        outW = static_cast<int>(std::ceil(float(inW) / strideW));

        int padH = std::max(0, (outH - 1) * strideH + kernelH - inH);
        int padW = std::max(0, (outW - 1) * strideW + kernelW - inW);

        padTop = padH / 2;
        padLeft = padW / 2;
    }
    else
    {
        outH = (inH - kernelH) / strideH + 1;
        outW = (inW - kernelW) / strideW + 1;
    }

    Tensor<float, 4> output(batch, outH, outW, channels);
    output.setZero();

    for (int b = 0; b < batch; ++b)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int oy = 0; oy < outH; ++oy)
            {
                for (int ox = 0; ox < outW; ++ox)
                {
                    float maxVal = -std::numeric_limits<float>::infinity();

                    for (int ky = 0; ky < kernelH; ++ky)
                    {
                        for (int kx = 0; kx < kernelW; ++kx)
                        {
                            int iy = oy * strideH + ky - padTop;
                            int ix = ox * strideW + kx - padLeft;

                            if (iy >= 0 && iy < inH && ix >= 0 && ix < inW)
                            {
                                float val = input(b, iy, ix, c);
                                if (val > maxVal)
                                    maxVal = val;
                            }
                        }
                    }
                    output(b, oy, ox, c) = maxVal;
                }
            }
        }
    }

    return output;
}

// Max pooling 1D
template <typename T>
Eigen::Tensor<T, 3> maxPool1D(const Eigen::Tensor<T, 3> &input,
                                 int kernel_size,
                                 int stride,
                                 bool same_padding = false)
{
    #ifdef DEBUG
    std::cout << "maxPool1D input dims:" << input.dimensions() << std::endl;
    #endif
    const int batch = input.dimension(0);
    const int inW = input.dimension(1);
    const int channels = input.dimension(2);

    int outW;
    int padLeft = 0;

    if (same_padding)
    {
        outW = static_cast<int>(std::ceil(float(inW) / stride));

        int padW = std::max(0, (outW - 1) * stride + kernel_size - inW);

        padLeft = padW / 2;
    }
    else
    {
        outW = (inW - kernel_size) / stride + 1;
    }

    Eigen::Tensor<T, 3> output(batch, outW, channels);
    output.setZero();

    for (int b = 0; b < batch; ++b)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int ox = 0; ox < outW; ++ox)
            {
                T maxVal = -std::numeric_limits<T>::infinity();

                for (int kx = 0; kx < kernel_size; ++kx)
                {
                    int ix = ox * stride + kx - padLeft;

                    if (ix >= 0 && ix < inW)
                    {
                        T val = input(b, ix, c);
                        if (val > maxVal)
                            maxVal = val;
                    }
                }
                output(b, ox, c) = maxVal;
            }
        }
    }

    return output;
}

template <typename T, int InputRank, int OutputRank>
Eigen::Tensor<T, OutputRank> flatten(const Eigen::Tensor<T, InputRank> &input, int start_dim = 1, int end_dim = -1)
{
    // Adjust negative end_dim
    if (end_dim < 0)
        end_dim += InputRank;
    #ifdef DEBUG
    std::cout << "flatten input rank:" << InputRank << std::endl;
    static_assert(InputRank >= 2, "flatten: InputRank must be >= 2");
    if (start_dim < 0 || start_dim >= InputRank || end_dim < 0 || end_dim >= InputRank || start_dim > end_dim)
        throw std::invalid_argument("flatten: start_dim or end_dim out of range");
    #endif
    // Compute new dimensions
    Eigen::array<Eigen::Index, OutputRank> newDims;
    int out_idx = 0;
    for (int i = 0; i < start_dim; ++i)
        newDims[out_idx++] = input.dimension(i);
    Eigen::Index flattened_size = 1;
    for (int i = start_dim; i <= end_dim; ++i)
        flattened_size *= input.dimension(i);
    newDims[out_idx++] = flattened_size;
    for (int i = end_dim + 1; i < InputRank; ++i)
        newDims[out_idx++] = input.dimension(i);
    return reshapeTensor<T, InputRank, OutputRank>(input, newDims);
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
    #ifdef DEBUG
    std::cout << "Conv input rank:" << InputRank << ", kernel rank:" << KernelRank << ", SpatialRank:" << SpatialRank << std::endl;
    static_assert(InputRank == SpatialRank + 1, "InputRank must be SpatialRank+1 (channels + spatial dims).");
    static_assert(OutputRank == SpatialRank + 1, "OutputRank must be SpatialRank+1 (filters + spatial dims).");
    static_assert(StrideRank == SpatialRank, "StrideRank must equal the number of spatial dimensions (SpatialRank).");
    #endif

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
    #ifdef DEBUG
    std::cout << "Conv C_in: " << C_in << ", C_out: " << C_out << ", kernel_C_in: " << kernel_C_in << std::endl;
    assert(kernel_C_in == C_in && "kernel second dimension must match input channels");
    #endif

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

#endif