---
layout: default
title: Examples
---

# Complete Examples

Working examples demonstrating how to use EmbeddedTorch.

## Example 1: Simple Classification Network

A basic fully connected network for classification.

```python
from layers import EmbaeddableModel, LinearLayer, flattenLayer
from operitions import reluLayer
from cpp import cpp_code, write_dep
import torch

# Create model
model = EmbaeddableModel(torch.float32)

# Add layers
model.add_layer(LinearLayer(784, 256, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(LinearLayer(256, 128, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(LinearLayer(128, 10, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))

# Generate C++ code
write_dep()
with open("out/classifier.cpp", "w", encoding="utf-8") as f:
    f.write(cpp_code(model.list))

print("C++ code generated in out/classifier.cpp")
```

## Example 2: CNN for Sequential Data

A convolutional neural network for 1D sequences.

```python
from layers import EmbaeddableModel, Conv1dLayer, MaxPool1dLayer, LinearLayer, flattenLayer
from operitions import reluLayer
from cpp import cpp_code, write_dep
import torch

# Create CNN model
model = EmbaeddableModel(torch.float32)

# Convolutional block
model.add_layer(Conv1dLayer(2, 4, 1, 1, (0, 1), dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(MaxPool1dLayer((2, 2), dtype=torch.float32))

# Another convolutional block
model.add_layer(Conv1dLayer(4, 8, 3, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(MaxPool1dLayer((2, 2), dtype=torch.float32))

# Flatten
model.add_layer(flattenLayer(4, dtype=torch.float32))

# Fully connected layers
model.add_layer(LinearLayer(32, 16, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(LinearLayer(16, 2, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))

# Generate C++
write_dep()
with open("out/cnn1d.cpp", "w", encoding="utf-8") as f:
    f.write(cpp_code(model.list))
```

## Example 3: Regression Model

A deep network for regression tasks.

```python
from layers import EmbaeddableModel, LinearLayer
from cpp import cpp_code, write_dep
import torch

# Create regression model
model = EmbaeddableModel(torch.float32)

# Deep layers
model.add_layer(LinearLayer(10, 64, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(LinearLayer(64, 32, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(LinearLayer(32, 16, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(LinearLayer(16, 1, dtype=torch.float32))

# No activation on output for regression
# Output layer directly gives prediction

# Generate C++
write_dep()
with open("out/regression.cpp", "w", encoding="utf-8") as f:
    f.write(cpp_code(model.list))
```

## Example 4: Multi-Task Network

A network with multiple outputs using model composition.

```python
from layers import EmbaeddableModel, LinearLayer, models_col
from operitions import reluLayer
from cpp import cpp_code, write_dep
import torch

# Create shared encoder
encoder = EmbaeddableModel(torch.float32)
encoder.add_layer(LinearLayer(128, 64, dtype=torch.float32))
encoder.add_layer(reluLayer(dtype=torch.float32))
encoder.add_layer(LinearLayer(64, 32, dtype=torch.float32))
encoder.add_layer(reluLayer(dtype=torch.float32))

# Create task-specific heads
head1 = EmbaeddableModel(torch.float32)
head1.add_layer(LinearLayer(32, 10, dtype=torch.float32))
head1.add_layer(reluLayer(dtype=torch.float32))

head2 = EmbaeddableModel(torch.float32)
head2.add_layer(LinearLayer(32, 5, dtype=torch.float32))
head2.add_layer(reluLayer(dtype=torch.float32))

# Combine models
model = models_col([encoder, head1])

# Generate C++
write_dep()
with open("out/multitask.cpp", "w", encoding="utf-8") as f:
    f.write(cpp_code(model.list))
```

## Example 5: With Batch Normalization

A network using normalization.

```python
from layers import EmbaeddableModel, LinearLayer, batchNorm1dLayer
from operitions import reluLayer
from cpp import cpp_code, write_dep
import torch

model = EmbaeddableModel(torch.float32)

# Linear layer with batch norm
model.add_layer(LinearLayer(128, 64, dtype=torch.float32))
model.add_layer(batchNorm1dLayer(64, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))

model.add_layer(LinearLayer(64, 32, dtype=torch.float32))
model.add_layer(batchNorm1dLayer(32, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))

model.add_layer(LinearLayer(32, 10, dtype=torch.float32))

# Generate C++
write_dep()
with open("out/batch_norm.cpp", "w", encoding="utf-8") as f:
    f.write(cpp_code(model.list))
```

## Example 6: Custom Function Integration

Creating a custom model with various operations.

```python
from layers import EmbaeddableModel, LinearLayer
from operitions import reluLayer, softmaxLayer
from cpp import cpp_code, write_dep
import torch

# Build a classifier
model = EmbaeddableModel(torch.float32)

# Feature extraction
model.add_layer(LinearLayer(784, 512, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(LinearLayer(512, 256, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))

# Classification head
model.add_layer(LinearLayer(256, 10, dtype=torch.float32))
model.add_layer(softmaxLayer(dim=1))

# Generate C++
write_dep()
with open("out/custom.cpp", "w", encoding="utf-8") as f:
    f.write(cpp_code(model.list))
```

## Running the Examples

1. Save any example code to a Python file (e.g., `example.py`)

2. Run the script:
```bash
python example.py
```

3. Check the generated C++ code in the `out/` directory

4. Compile the C++ code (make sure you have Eigen library):
```bash
cd out
g++ -std=c++11 -I/path/to/eigen code.cpp
```

## Tips for Customization

### Adjusting Layer Sizes

```python
# Change input/output dimensions
layer = LinearLayer(100, 50, dtype=torch.float32)  # 100 -> 50 features
```

### Changing Activation Functions

```python
from operitions import reluLayer, sigmoidLayer, tanhLayer

# Use different activations
model.add_layer(reluLayer(dtype=torch.float32))      # Fast
model.add_layer(sigmoidLayer())                       # Bounded
model.add_layer(tanhLayer())                          # Zero-centered
```

### Working with Different Data Types

```python
# Use float32 (default)
model = EmbaeddableModel(torch.float32)

# Or use float64 for higher precision
model = EmbaeddableModel(torch.float64)
```

## Troubleshooting Examples

### Shape Mismatch

If you get dimension errors, check layer compatibility:

```python
# Good: 128 -> 64 -> 32
model.add_layer(LinearLayer(128, 64, dtype=torch.float32))
model.add_layer(LinearLayer(64, 32, dtype=torch.float32))

# Bad: dimensions don't match
# model.add_layer(LinearLayer(128, 64))
# model.add_layer(LinearLayer(100, 32))  # Wrong input size!
```

### Dtype Consistency

Ensure consistent data types:

```python
# Good: all layers use same dtype
model = EmbaeddableModel(torch.float32)
layer = LinearLayer(10, 5, dtype=torch.float32)

# Bad: mixing dtypes
# layer = LinearLayer(10, 5, dtype=torch.float64)  # Mismatch!
```

## Next Steps

- Modify these examples for your specific use case
- Experiment with different architectures
- Read the [Layers Reference]({{ "layers" | relative_url }}) documentation
- Check out [Operations]({{ "operations" | relative_url }}) for more functions

