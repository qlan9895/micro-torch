# micro-torch #

A neural network library with torch like syntax. 

## Features

- Tensor class with built in auto-grad support. Able to build simple neural networks with arbitrary width and depth without worrying about back-propagation.
- Base module able to track updateable parameters in a neural network.
- A simple batch gradient descend optimizer.

## Implementation details

#### Tensor

Data structure that represent weight matrices, biaes vectors, data batch etc. Has following attributes:
- `data`: Actual data. Normally an `np.array`.
- `require_grad`: a boolean value that determines whether gradient flow pass through this Tensor.
- `grad`: The gradient $\frac{dL}{d_{data}}$, computed during backward pass. a `np.array`.
- `op`: A string representation of how this tensor is computed. Used for short circuiting certain gradient computation during back-propagation (cross entropy loss with softmax input).
- `_prev`: Point to tensors by which it is constructed. i.e. if `A = B + C`, then `A._prev = (B, C)`.
- `_backward`: a function that computes `Tensor.grad`, initialized during forward pass and get called during backward pass.
- `_is_leaf`: whether Tensor._prev is empty.

Basic operation like addition, multiplication, subtraction and matrix multiply are supported. Each operation have a `_backward` function that compute gradients of previous tensors. 

`Tensor.backward()` gets called after the loss is computed. Forward pass will construct a computational graph (topology) based on the order of operations, which will be traversed backward and call `_backward` on each Tensor to compute gradient. 

#### Module
Base class for a basic linear layer, a neural network etc. Can track its parameters and submodules. It has following attributes:
- `_params`: a dict that tracks current parameters in the module.
- `_modules`: a dict that tracks submodules, it happens when 
- `get_params()` will return a list of Tensors that are updateable parameters.

#### F
Stateless functions. Operates on Tensor and returns a Tensor. All activation function and loss function belong to this. Also have `_backward` that get called during backward pass

#### Linear
Represent a linear layer in a neural network. A subclass of Module. Will initialize a weight Tensor with dimension (output_dim, input_dim). Have a `forward` method. 

#### Optimizer
Mount on Module class. Have two inputs: `lr` and `parameters`. Can reset gradients and update gradients. 

## Usage
See `\examples`

## To do:
- broadcast support: for __sub__
- Metal acceleration
- More Tensors operations: `concat`, `flatten` etc.
