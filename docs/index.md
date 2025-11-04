---
layout: default
title: Home
permalink: /
---

# EmbeddedTorch

**Convert your PyTorch models to embedded C++ code**

EmbeddedTorch is a powerful Python library that allows you to convert PyTorch neural network models into optimized C++ code for deployment in embedded systems, microcontrollers, and resource-constrained environments.

## Key Features

- 🚀 **Easy Conversion**: Convert PyTorch models to C++ with simple Python API
- 📦 **Comprehensive Layers**: Support for linear, convolutional, pooling, normalization, and activation layers
- ⚙️ **Operations**: Rich set of mathematical and tensor operations
- 🎯 **Type-Safe**: Support for different data types (float32, float64, etc.)
- 🔧 **Flexible**: Custom layer definitions and model composition support
- 🏗️ **Production-Ready**: Generate clean, optimized C++ code

## Quick Start

```python
from embeddedtorch.layers import EmbaeddableModel, LinearLayer
import torch

# Create a model
model = EmbaeddableModel(torch.float32)
model.add_layer(LinearLayer(10, 5, dtype=torch.float32))
model.add_layer(LinearLayer(5, 2, dtype=torch.float32))

# Generate C++ code
write_dep()
with open("out/code.cpp", "w", encoding="utf-8") as f:
    print(cpp_code(model.list), file=f)
```

## Documentation

- **[Getting Started]({{ "getting-started" | relative_url }})** - Installation and basic usage
- **[API Reference]({{ "api-reference" | relative_url }})** - Detailed API documentation
- **[Layers]({{ "layers" | relative_url }})** - Available layer types and usage
- **[Operations]({{ "operations" | relative_url }})** - Mathematical operations
- **[Examples]({{ "examples" | relative_url }})** - Complete working examples

## Get Started

Ready to convert your PyTorch models to C++? Check out the [Getting Started]({{ "getting-started" | relative_url }}) guide.

## Installation

```bash
pip install embeddedtorch
```

## License

MIT License - See LICENSE file for details