import numpy as np
from core.base import Tensor

def test_add():
    x = Tensor(np.array([1.0, 2.0]),require_grad=True)
    y = Tensor(np.array([3.0, 4.0]), require_grad=True)
    c = x + y
    c.backward()
    assert isinstance(c, Tensor)
    assert c.op == "__add__"
    assert np.allclose(c.data, np.array([4.0, 6.0]))
    assert x.grad is not None and np.allclose(x.grad, np.array([1.0, 1.0]))
    assert y.grad is not None and np.allclose(y.grad, np.array([1.0, 1.0]))

def test_mul():
    a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), require_grad=True)
    b = Tensor(np.array([[1.0, 3.0], [2.0, 1.0]]), require_grad=True)
    c = a * b
    c.backward()
    assert c.op == "__mul__"
    assert np.allclose(c.data, np.array([[1.0, 0.0],[0.0, 1.0]]))
    assert a.grad is not None and np.allclose(a.grad, np.array([[1.0, 3.0], [2.0, 1.0]]))
    assert b.grad is not None and np.allclose(b.grad, np.array([[1.0, 0.0], [0.0, 1.0]]))

def test_matmal():
    a = Tensor(np.array([[1.0, 1.0], [2.0, 3.0], [-1.0, 1.0]]), require_grad=True) # a 3x2 matrix
    b = Tensor(np.array([[1.0, 0.0, 3.0], [2.0, 6.0, 1.0]]), require_grad=True) # a 2x3 matrix
    c = a @ b
    c.backward()
    assert c.op == "__matmul__"
    assert np.allclose(c.data, np.array([[3.0, 6.0, 4.0], [8.0, 18.0, 9.0], [1.0, 6.0, -2.0]]))
    assert a.grad is not None and a.grad.shape == (3, 2)
    assert b.grad is not None and b.grad.shape == (2, 3)

