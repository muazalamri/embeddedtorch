---
layout: default
title: Layers Reference
---

# Layers Reference

Complete reference for all available layer types in EmbeddedTorch.

## Linear Layers

### LinearLayer

Fully connected (dense) linear layer for neural networks.

**Constructor:**
```python
LinearLayer(in_features, out_features, bias=True, dtype=torch.float32)
```

**Parameters:**
- `in_features` (int): Number of input features
- `out_features` (int): Number of output features  
- `bias` (bool): Whether to use bias term (default: True)
- `dtype` (torch.dtype): Data type (default: torch.float32)

**Example:**
```python
layer = LinearLayer(128, 64, dtype=torch.float32)
```

## Convolutional Layers

### Conv1dLayer

1D convolution for sequence data.

**Constructor:**
```python
Conv1dLayer(in_channels, out_channels, kernel_size, stride=1, padding=0, 
           dilation=1, groups=1, bias=True, dtype=torch.float32)
```

**Example:**
```python
layer = Conv1dLayer(in_channels=2, out_channels=4, kernel_size=3, 
                   stride=1, padding=1, dtype=torch.float32)
```

### Conv2dLayer

2D convolution for image data.

**Constructor:**
```python
Conv2dLayer(in_channels, out_channels, kernel_size, stride=1, padding=0,
            dilation=1, groups=1, bias=True, dtype=torch.float32)
```

**Example:**
```python
layer = Conv2dLayer(in_channels=3, out_channels=16, kernel_size=3,
                   stride=1, padding=1, dtype=torch.float32)
```

### Conv3dLayer

3D convolution for volumetric data.

**Constructor:**
```python
Conv3dLayer(in_channels, out_channels, kernel_size, stride=1, padding=0,
            dilation=1, groups=1, bias=True, dtype=torch.float32)
```

## Pooling Layers

### MaxPool1dLayer

1D maximum pooling operation.

**Constructor:**
```python
MaxPool1dLayer(kernel_size, stride=None, padding=0, dilation=1, 
               return_indices=False, ceil_mode=False, dtype=torch.float32)
```

**Example:**
```python
layer = MaxPool1dLayer(kernel_size=(2, 2), dtype=torch.float32)
```

### MaxPool2dLayer

2D maximum pooling operation.

**Constructor:**
```python
MaxPool2dLayer(kernel_size=(2, 2), stride=(2, 2), dtype=torch.float32)
```

### MaxPool3dLayer

3D maximum pooling operation.

**Constructor:**
```python
MaxPool3dLayer(kernel_size, stride=None, padding=0, dilation=1,
               return_indices=False, ceil_mode=False, dtype=torch.float32)
```

## Normalization Layers

### Batch Normalization

#### batchNorm1dLayer

**Constructor:**
```python
batchNorm1dLayer(num_features, eps=1e-5, momentum=0.1, affine=True, dtype=torch.float32)
```

#### batchNorm2dLayer

**Constructor:**
```python
batchNorm2dLayer(num_features, eps=1e-5, momentum=0.1, affine=True, dtype=torch.float32)
```

#### batchNorm3dLayer

**Constructor:**
```python
batchNorm3dLayer(num_features, eps=1e-5, momentum=0.1, affine=True, dtype=torch.float32)
```

### Layer Normalization

#### LayerNorm1dLayer

**Constructor:**
```python
LayerNorm1dLayer(normalized_shape, eps=1e-5, elementwise_affine=True, dtype=torch.float32)
```

#### LayerNorm2dLayer

**Constructor:**
```python
LayerNorm2dLayer(normalized_shape, eps=1e-5, elementwise_affine=True, dtype=torch.float32)
```

#### LayerNorm3dLayer

**Constructor:**
```python
LayerNorm3dLayer(normalized_shape, eps=1e-5, elementwise_affine=True, dtype=torch.float32)
```

## Reshaping Layers

### flattenLayer

Flatten input tensor.

**Constructor:**
```python
flattenLayer(inputRank, outputRank=2, dtype=torch.float32, 
             start_dim=1, end_dim=-1)
```

**Example:**
```python
# Flatten a 4D tensor (B, C, H, W) to 2D (B, C*H*W)
layer = flattenLayer(inputRank=4, outputRank=2)

# Flatten a 3D tensor (B, C, L) to 2D (B, C*L)
layer = flattenLayer(inputRank=3, outputRank=2)
```

### reshapeLayer

Reshape input to specified shape.

**Constructor:**
```python
reshapeLayer(shape, dtype=torch.float32)
```

**Example:**
```python
layer = reshapeLayer(shape=(1, 256))
```

## Upsampling Layers

### UpSample1dLayer

1D upsampling operation.

**Constructor:**
```python
UpSample1dLayer(size=None, scale_factor=None, mode='nearest', 
                align_corners=None, dtype=torch.float32)
```

### UpSample2dLayer

2D upsampling operation.

**Constructor:**
```python
UpSample2dLayer(size=None, scale_factor=None, mode='nearest',
                align_corners=None, dtype=torch.float32)
```

### UpSample3dLayer

3D upsampling operation.

**Constructor:**
```python
UpSample3dLayer(size=None, scale_factor=None, mode='nearest',
                align_corners=None, dtype=torch.float32)
```

## Special Layers

### dropoutLayer

Dropout for regularization.

**Constructor:**
```python
dropoutLayer(p=0.5, dtype=torch.float32)
```

### EmbeddingLayer

Embedding layer for categorical data.

**Constructor:**
```python
EmbeddingLayer(num_embeddings, embedding_dim, padding_idx=None,
               max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
               sparse=False, dtype=torch.float32)
```

## Model Composition

### models_col

Combine models in series (sequential).

**Constructor:**
```python
models_col(models: list)
```

**Example:**
```python
model1 = EmbaeddableModel(torch.float32)
model2 = EmbaeddableModel(torch.float32)

combined = models_col([model1, model2])
```

### models_row

Combine models in parallel with splitting.

**Constructor:**
```python
models_row(models: list, split_points: list[int], split_dim=1)
```

**Example:**
```python
model1 = EmbaeddableModel(torch.float32)
model2 = EmbaeddableModel(torch.float32)

combined = models_row([model1, model2], split_points=[4, 4], split_dim=1)
```

## Layer Execution Order

Layers are executed in the order they are added to the model:

```python
model = EmbaeddableModel(torch.float32)
model.add_layer(layer1)  # Executes first
model.add_layer(layer2)  # Executes second
model.add_layer(layer3)  # Executes third
```

## Best Practices

1. **Consistent dtype**: Use the same dtype for all layers in a model
2. **Layer compatibility**: Ensure input/output shapes match between consecutive layers
3. **Memory considerations**: Be mindful of tensor dimensions for embedded systems
4. **Testing**: Test your PyTorch model before conversion

