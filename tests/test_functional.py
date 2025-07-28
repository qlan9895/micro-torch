import numpy as np
from core.functional import F
from core.base import Tensor

def test_sigmoid():
    v = Tensor(np.array([[1.0, 1.0], [0.0, 1.0]]), require_grad=True)
    v_final = F.sigmoid(v)
    v_final.backward()
    assert np.allclose(v.grad, v_final.data * (1 - v_final.data))

def test_relu():
    v = Tensor(np.array([[1.0, 1.0], [0.0, 1.0]]), require_grad=True)
    v_final = F.relu(v)
    v_final.backward()
    assert np.allclose(v_final.data, np.array([[1.0, 1.0], [0.0, 1.0]]))
    assert np.allclose(v.grad, np.array([[1.0, 1.0],[0.0, 1.0]]))

def test_softmax():
    v = Tensor(np.array([[1.0, 0.5, 2.5],[3.0, 0.0, 1.0],[2.5, 1.0, 4.0]]), require_grad=True)
    v_final = F.softmax(v)
    v_final.backward()
    assert np.allclose(np.sum(v_final.data, axis=0), np.array([[1.0, 1.0, 1.0]]))
    print(v.grad)