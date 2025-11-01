---
layout: default
title: API Reference
---

# API Reference

Complete API documentation for EmbeddedTorch.

## Core Classes

### `EmbaeddableModel`

Main model class that holds a collection of layers.

```python
from layers import EmbaeddableModel
import torch

model = EmbaeddableModel(dtype=torch.float32)
```

#### Constructor

- **Parameters**:
  - `dtype` (torch.dtype): Data type for the model (default: torch.float32)

#### Methods

##### `add_layer(layer)`

Add a layer to the model.

- **Parameters**:
  - `layer` (torch.nn.Module): A layer instance
- **Returns**: None

##### `forward(x)`

Perform forward pass through the model.

- **Parameters**:
  - `x` (torch.Tensor): Input tensor
- **Returns**: Output tensor

## Layer Classes

### LinearLayer

Fully connected linear layer.

```python
from layers import LinearLayer

layer = LinearLayer(in_features=10, out_features=5, bias=True, dtype=torch.float32)
```

#### Constructor Parameters

- `in_features` (int): Number of input features
- `out_features` (int): Number of output features
- `bias` (bool): Whether to include bias term (default: True)
- `dtype` (torch.dtype): Data type (default: torch.float32)

#### Methods

##### `forward(x)`

Apply linear transformation.

##### `to_cpp(layer_num)`

Convert to C++ code.

- **Parameters**:
  - `layer_num` (int): Layer index for naming
- **Returns**: Tuple of (init_code, set_code, forward_code)

### Conv1dLayer

1D convolutional layer.

```python
from layers import Conv1dLayer

layer = Conv1dLayer(
    in_channels=2,
    out_channels=4,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    dtype=torch.float32
)
```

#### Constructor Parameters

- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels
- `kernel_size` (int): Size of the convolution kernel
- `stride` (int): Stride of the convolution (default: 1)
- `padding` (int or tuple): Zero-padding (default: 0)
- `bias` (bool): Whether to include bias (default: True)
- `dtype` (torch.dtype): Data type (default: torch.float32)

### Conv2dLayer

2D convolutional layer.

```python
from layers import Conv2dLayer

layer = Conv2dLayer(
    in_channels=3,
    out_channels=16,
    kernel_size=(3, 3),
    stride=(1, 1),
    padding=(1, 1),
    bias=True,
    dtype=torch.float32
)
```

### MaxPool1dLayer

1D maximum pooling layer.

```python
from layers import MaxPool1dLayer

layer = MaxPool1dLayer(
    kernel_size=2,
    stride=2,
    padding=0,
    dtype=torch.float32
)
```

### flattenLayer

Flatten layer for reshaping tensors.

```python
from layers import flattenLayer

layer = flattenLayer(
    inputRank=4,
    outputRank=2,
    dtype=torch.float32,
    start_dim=1,
    end_dim=-1
)
```

#### Constructor Parameters

- `inputRank` (int): Input tensor rank
- `outputRank` (int): Output tensor rank
- `start_dim` (int): First dim to flatten (default: 1)
- `end_dim` (int): Last dim to flatten (default: -1)
- `dtype` (torch.dtype): Data type (default: torch.float32)

## Operations

### Activation Functions

Available in `operitions.py`:

#### reluLayer

Rectified Linear Unit activation.

```python
from operitions import reluLayer

layer = reluLayer(dtype=torch.float32)
```

#### sigmoidLayer

Sigmoid activation function.

#### tanhLayer

Hyperbolic tangent activation.

#### softmaxLayer

Softmax activation.

```python
from operitions import softmaxLayer

layer = softmaxLayer(dim=1)
```

### Mathematical Operations

#### Add, Subtract, Multiply, Divide

```python
from operitions import addLayer, subLayer, mulLayer, divLayer

add_op = addLayer()
sub_op = subLayer()
mul_op = mulLayer()
div_op = divLayer()
```

#### Reduce Operations

```python
from operitions import meanLayer, sumLayer, maxLayer, minLayer

mean_op = meanLayer(dim=1, keepdim=False)
sum_op = sumLayer(dim=0, keepdim=True)
```

## Utility Functions

### `tensor2cpp(tensor, dtype)`

Convert PyTorch tensor to C++ array representation.

- **Parameters**:
  - `tensor` (torch.Tensor): PyTorch tensor
  - `dtype` (type): Python dtype
- **Returns**: str

### `cpp_code(layers)`

Generate C++ code for a list of layers.

- **Parameters**:
  - `layers` (list): List of layer objects
- **Returns**: str (complete C++ code)

### `write_dep()`

Write dependency files to output directory.

- **Returns**: None

## Example Usage

```python
from layers import EmbaeddableModel, LinearLayer, Conv1dLayer
from operitions import reluLayer
from cpp import cpp_code, write_dep
import torch

# Create model
model = EmbaeddableModel(torch.float32)

# Add layers
model.add_layer(Conv1dLayer(2, 4, 3, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(flattenLayer(4))
model.add_layer(LinearLayer(16, 10, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(LinearLayer(10, 1, dtype=torch.float32))

# Generate C++
write_dep()
with open("out/code.cpp", "w") as f:
    f.write(cpp_code(model.list))
```

