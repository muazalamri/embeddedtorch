template <typename dtype>
dtype add(dtype x1, dtype x2) {
    for (int i = 0; i < sizeof(x1)/sizeof(x1[0]); i++)
    {
        x1[i] = x1[i] + x2[i];
    }
    
};
template <typename dtype>
dtype sub(dtype x1, dtype x2) {
    for (int i = 0; i < sizeof(x1)/sizeof(x1[0]); i++)
    {
        x1[i] = x1[i] - x2[i];
    }
};
int calc_index(int *position, int *strides, int ndim)
{
    int index = 0;
    for (int i = 0; i < ndim; i++)
    {
        index += position[i] * strides[i];
    }
    return index;
}
void compute_strides(int* shape, int* strides, int ndim) {
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}
template <typename dtype>
dtype mul(dtype x1, dtype x2) {
    int* shape[]=x1.shape;
    int* strides[]=x1.strides;
    int* posion[]=new int[sizeof(shape)/sizeof(shape[0])];
    
};
template <typename dtype>
dtype div(dtype x1, dtype x2) {};
template <typename dtype>
dtype power(dtype x1, dtype x2) {};
template <typename dtype>
dtype dtypemul(dtype x1, dtype x2) {};
template <typename dtype>
dtype cat(dtype x1, dtype x2, int dim) {};
template <typename dtype>
dtype stack(dtype x1, dtype x2, int dim) {};
template <typename dtype>
dtype mean(dtype x, int dim, bool keepdim) {};
template <typename dtype>
dtype sum(dtype x, int dim, bool keepdim) {};
template <typename dtype>
dtype max(dtype x, int dim, bool keepdim) {};
template <typename dtype>
dtype min(dtype x, int dim, bool keepdim) {};
template <typename dtype>
dtype relu(dtype x) {};
template <typename dtype>
dtype sigmoid(dtype x) {};
template <typename dtype>
dtype tanh(dtype x) {};
template <typename dtype>
dtype softmax(dtype x, int dim) {};
template <typename dtype>
dtype log(dtype x) {};
template <typename dtype>
dtype exp(dtype x) {};
template <typename dtype>
dtype sqrt(dtype x) {};
template <typename dtype>
dtype abs(dtype x) {};
