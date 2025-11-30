# Neural Network Weights Directory

This directory should contain the neural network weights and scaling parameters for the NN-based turbulence models.

## File Format

### For `nn_mlp` model (scalar eddy viscosity)

```
layer0_W.txt    # Weight matrix for layer 0 (space-separated, row-major)
layer0_b.txt    # Bias vector for layer 0 (one value per line)
layer1_W.txt    # Weight matrix for layer 1
layer1_b.txt    # Bias vector for layer 1
...
input_means.txt # Input feature means (one per line)
input_stds.txt  # Input feature standard deviations (one per line)
```

### For `nn_tbnn` model (TBNN anisotropy)

Same format as above, but the output dimension should match the number of tensor basis functions (typically 4 for 2D).

## Exporting Weights from Python

### PyTorch Example

```python
import torch
import numpy as np

# Load your trained model
model = torch.load('model.pth')
model.eval()

# Export each layer
for i, layer in enumerate(model.layers):
    W = layer.weight.detach().cpu().numpy()  # Shape: (out_features, in_features)
    b = layer.bias.detach().cpu().numpy()    # Shape: (out_features,)
    
    # Save in space-separated format
    np.savetxt(f'layer{i}_W.txt', W, fmt='%.16e')
    np.savetxt(f'layer{i}_b.txt', b, fmt='%.16e')

# Export input scaling (if used during training)
np.savetxt('input_means.txt', feature_means, fmt='%.16e')
np.savetxt('input_stds.txt', feature_stds, fmt='%.16e')
```

### TensorFlow/Keras Example

```python
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('model.h5')

# Export each dense layer
layer_idx = 0
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        W, b = layer.get_weights()  # W shape: (in_features, out_features)
        W = W.T  # Transpose to match C++ convention (out, in)
        
        np.savetxt(f'layer{layer_idx}_W.txt', W, fmt='%.16e')
        np.savetxt(f'layer{layer_idx}_b.txt', b, fmt='%.16e')
        layer_idx += 1
```

## Example File Contents

**layer0_W.txt** (3 inputs, 2 outputs):
```
1.234567e-01 -2.345678e-01  3.456789e-01
4.567890e-01  5.678901e-01 -6.789012e-01
```

**layer0_b.txt**:
```
7.890123e-02
-8.901234e-02
```

## Feature Ordering

Features should match the order expected by the C++ code. For scalar `nu_t` model:

0. Normalized strain rate magnitude
1. Normalized rotation rate magnitude  
2. Normalized wall distance
3. Strain-rotation ratio
4. Local Reynolds number
5. Normalized velocity magnitude

For TBNN model, features include tensor invariants as defined in `features.cpp`.


