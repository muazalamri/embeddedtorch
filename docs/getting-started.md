---
layout: default
title: Getting Started
---

# Getting Started with EmbeddedTorch

This guide will help you get started with EmbeddedTorch and convert your first PyTorch model to C++.

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/embeddedtorch.git
cd embeddedtorch

# Install in development mode
pip install -e .
```

## Your First Conversion

Let's create a simple neural network and convert it to C++.

### Step 1: Import Required Modules

```python
from embeddedtorch.layers import EmbaeddableModel, LinearLayer
from embeddedtorch.operitions import reluLayer
from embeddedtorch.cpp import cpp_code, write_dep
import torch
```

### Step 2: Create a Model

```python
# Create a model instance
model = EmbaeddableModel(torch.float32)

# Add layers to your model
model.add_layer(LinearLayer(10, 8, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(LinearLayer(8, 4, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(LinearLayer(4, 1, dtype=torch.float32))
```

### Step 3: Generate C++ Code

```python
# Write dependencies
write_dep()

# Generate and save C++ code
with open("out/code.cpp", "w", encoding="utf-8") as f:
    print(cpp_code(model.list), file=f)
```

### Step 4: Review the Output

The generated C++ code will be in `out/code.cpp`. This file contains:
- Tensor definitions
- Weight and bias initialization
- Forward pass implementation

## Understanding the Output Structure

### Generated Files

```
out/
├── code.cpp       # Main C++ code with model inference
└── func.hpp       # Helper functions for tensor operations
```

### Code Structure

The generated C++ code includes:

1. **Tensor Initialization**: Declaration of weight and bias tensors
2. **Weight Loading**: Setting values from trained model
3. **Forward Pass**: Sequential layer execution
4. **Output**: Final prediction tensor

## A More Complex Example

### Using Convolutional and Pooling Layers

```python
from embeddedtorch.layers import EmbaeddableModel, LinearLayer, Conv1dLayer, MaxPool1dLayer, flattenLayer
from embeddedtorch.operitions import reluLayer

# Create a CNN model
model = EmbaeddableModel(torch.float32)

# Convolutional layer
model.add_layer(Conv1dLayer(2, 2, 1, 1, (0, 1)))
model.add_layer(MaxPool1dLayer((2, 2)))

# Flatten
model.add_layer(flattenLayer(4))

# Fully connected layers
model.add_layer(LinearLayer(16, 8, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(LinearLayer(8, 4, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(LinearLayer(4, 2, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))
model.add_layer(LinearLayer(2, 1, dtype=torch.float32))
model.add_layer(reluLayer(dtype=torch.float32))

# Generate C++ code
write_dep()
with open("out/code.cpp", "w", encoding="utf-8") as f:
    print(cpp_code(model.list), file=f)
```

## Next Steps

- Learn about different [layer types]({{ "layers" | relative_url }})
- Explore [available operations]({{ "operations" | relative_url }})
- Check out more [examples]({{ "examples" | relative_url }})
- Read the [API reference]({{ "api-reference" | relative_url }})

## Troubleshooting

### Common Issues

**ImportError**: Make sure all dependencies are installed
```bash
pip install torch numpy
```

**File Not Found**: Ensure the `out/` directory exists
```bash
mkdir -p out
```

**Type Mismatch**: Ensure consistent dtype across all layers
```python
# Make sure all layers use the same dtype
model = EmbaeddableModel(torch.float32)  # Define dtype
layer = LinearLayer(10, 5, dtype=torch.float32)  # Match dtype
```

## Need Help?

- Check the [documentation]({{ "api-reference" | relative_url }})
- Review [examples]({{ "examples" | relative_url }})
- Open an issue on GitHub

