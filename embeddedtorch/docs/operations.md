---
layout: default
title: Operations Reference
---

# Operations Reference

Complete reference for all available operations in EmbeddedTorch.

## Activation Functions

Activation functions introduce non-linearity to the model.

### reluLayer

Rectified Linear Unit - most common activation function.

**Usage:**
```python
from operitions import reluLayer

layer = reluLayer(dtype=torch.float32)
```

**Properties:**
- Replaces negative values with zero
- Fast to compute
- May suffer from vanishing gradients

### sigmoidLayer

Sigmoid activation - outputs values between 0 and 1.

**Usage:**
```python
from operitions import sigmoidLayer

layer = sigmoidLayer()
```

**Properties:**
- Range: (0, 1)
- Smooth gradient
- Common for binary classification

### tanhLayer

Hyperbolic tangent activation.

**Usage:**
```python
from operitions import tanhLayer

layer = tanhLayer()
```

**Properties:**
- Range: (-1, 1)
- Zero-centered
- Smoother than sigmoid

### softmaxLayer

Softmax activation - normalizes outputs to probability distribution.

**Usage:**
```python
from operitions import softmaxLayer

layer = softmaxLayer(dim=1)
```

**Parameters:**
- `dim` (int): Dimension along which to compute softmax (default: 1)

**Properties:**
- Output sums to 1
- Used for multi-class classification
- Converts logits to probabilities

## Arithmetic Operations

### Binary Operations

Operations that take two input tensors.

#### addLayer

Element-wise addition.

```python
from operitions import addLayer

add_op = addLayer()
# forward: x1 + x2
```

#### subLayer

Element-wise subtraction.

```python
from operitions import subLayer

sub_op = subLayer()
# forward: x1 - x2
```

#### mulLayer

Element-wise multiplication.

```python
from operitions import mulLayer

mul_op = mulLayer()
# forward: x1 * x2
```

#### divLayer

Element-wise division.

```python
from operitions import divLayer

div_op = divLayer()
# forward: x1 / x2
```

#### powLayer

Element-wise power operation.

```python
from operitions import powLayer

pow_op = powLayer()
# forward: x1 ** x2
```

#### matmulLayer

Matrix multiplication.

```python
from operitions import matmulLayer

matmul_op = matmulLayer()
# forward: torch.matmul(x1, x2)
```

### Concatenation Operations

#### catLayer

Concatenate tensors along a dimension.

```python
from operitions import catLayer

layer = catLayer(dim=1)
# forward: torch.cat((x1, x2), dim=dim)
```

**Parameters:**
- `dim` (int): Dimension along which to concatenate (default: 1)

#### stackLayer

Stack tensors along a dimension.

```python
from operitions import stackLayer

layer = stackLayer(dim=1)
# forward: torch.stack((x1, x2), dim=dim)
```

**Parameters:**
- `dim` (int): Dimension along which to stack (default: 1)

## Reduction Operations

Operations that reduce tensor dimensions.

### meanLayer

Compute mean along specified dimension.

```python
from operitions import meanLayer

layer = meanLayer(dim=1, keepdim=False)
# forward: torch.mean(x, dim=dim, keepdim=keepdim)
```

**Parameters:**
- `dim` (int): Dimension to reduce (default: 1)
- `keepdim` (bool): Keep dimension in output (default: False)

### sumLayer

Compute sum along specified dimension.

```python
from operitions import sumLayer

layer = sumLayer(dim=0, keepdim=True)
# forward: torch.sum(x, dim=dim, keepdim=keepdim)
```

**Parameters:**
- `dim` (int): Dimension to reduce (default: 1)
- `keepdim` (bool): Keep dimension in output (default: False)

### maxLayer

Compute maximum along specified dimension.

```python
from operitions import maxLayer

layer = maxLayer(dim=1, keepdim=False)
# forward: torch.max(x, dim=dim, keepdim=keepdim).values
```

### minLayer

Compute minimum along specified dimension.

```python
from operitions import minLayer

layer = minLayer(dim=1, keepdim=False)
# forward: torch.min(x, dim=dim, keepdim=keepdim).values
```

## Element-wise Operations

### logLayer

Natural logarithm.

```python
from operitions import logLayer

log_op = logLayer()
# forward: torch.log(x)
```

### expLayer

Exponential function.

```python
from operitions import expLayer

exp_op = expLayer()
# forward: torch.exp(x)
```

### sqrtLayer

Square root.

```python
from operitions import sqrtLayer

sqrt_op = sqrtLayer()
# forward: torch.sqrt(x)
```

### absLayer

Absolute value.

```python
from operitions import absLayer

abs_op = absLayer()
# forward: torch.abs(x)
```

## Usage Examples

### Creating a Custom Model with Operations

```python
from layers import EmbaeddableModel, LinearLayer
from operitions import reluLayer, addLayer
import torch

# Create model
model = EmbaeddableModel(torch.float32)

# Add layers
model.add_layer(LinearLayer(10, 5, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(LinearLayer(5, 2, dtype=torch.float32))

# Apply operations in forward pass
# (Note: Operations are typically used in custom module definitions)
```

### Combining Operations

```python
from operitions import meanLayer, sumLayer, softmaxLayer

# Compute statistics
mean_op = meanLayer(dim=1, keepdim=True)
sum_op = sumLayer(dim=1, keepdim=True)
prob_op = softmaxLayer(dim=-1)
```

## Performance Considerations

1. **Activation functions**: ReLU is fastest, sigmoid/tanh are slower
2. **Reduction operations**: Can be memory-intensive on large tensors
3. **Binary operations**: Require compatible tensor shapes
4. **Matrix operations**: Use with care in embedded systems

## Best Practices

1. Use ReLU as default activation unless you need bounded output
2. Keep reduction dimensions to avoid unnecessary computation
3. Ensure tensor shape compatibility for binary operations
4. Profile your model to identify bottlenecks

