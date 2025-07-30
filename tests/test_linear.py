from microtorch.core.tensor import Tensor
import numpy as np
from microtorch.core.linear import Linear

def test_linear():
    x = np.array([[1.0, 1.0, 2.0], [0.5, 1.0, 1.0], [3.0, 1.0, 1.0]])
    l1 = Linear(4, 3)
    l2 = l1(x)
    assert len(l1._params) == 2
    print(l1._params)
    assert l2.data.shape == (4, 3)
    l2.backward()
    print(l1.W.grad)
    print(l1.b.grad)