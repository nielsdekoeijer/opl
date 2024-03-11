import opl
import pytest
import numpy as np

# ===================================================================
# scale
def test_scale_f32():
    # setup ref
    factor = 5.0
    size = 10
    inp = np.random.randn(size)
    ref = factor * inp
    
    # setup out
    out = opl.vecf(10)
    for i in range(size):
        out[i] = inp[i]
    opl.scale(factor, out)
    # test
    for i in range(size):
        assert out[i] == pytest.approx(ref[i])

def test_scale_f64():
    # setup ref
    factor = 5.0
    size = 10
    inp = np.random.randn(size)
    ref = factor * inp
    
    # setup out
    out = opl.vecd(10)
    for i in range(size):
        out[i] = inp[i]
    opl.scale(factor, out)
    
    # test
    for i in range(size):
        assert out[i] == pytest.approx(ref[i])

def test_scale_c32():
    # setup ref
    factor = 5.0
    size = 10
    inp = np.random.randn(size) + 1j * np.random.randn(size)
    ref = factor * inp
    
    # setup out
    out = opl.vecx(10)
    for i in range(size):
        out[i] = inp[i]
    opl.scale(factor, out)
    
    # test
    for i in range(size):
        assert out[i] == pytest.approx(ref[i])

def test_scale_c64():
    # setup ref
    factor = 5.0
    size = 10
    inp = np.random.randn(size) + 1j * np.random.randn(size)
    ref = factor * inp
    
    # setup out
    out = opl.vecz(10)
    for i in range(size):
        out[i] = inp[i]
    opl.scale(factor, out)
    
    # test
    for i in range(size):
        assert out[i] == pytest.approx(ref[i])

# ===================================================================
# dot
def test_dot_f32():
    # setup ref
    factor = 5.0
    size = 10
    inp_1 = np.random.randn(size)
    inp_2 = np.random.randn(size)
    ref = np.inner(inp_1, inp_2)
    
    # setup out
    opl_1 = opl.vecf(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecf(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    out = opl.dot(opl_1, opl_2)
    
    # test
    assert out == pytest.approx(ref)

def test_dot_f64():
    # setup ref
    factor = 5.0
    size = 10
    inp_1 = np.random.randn(size)
    inp_2 = np.random.randn(size)
    ref = np.inner(inp_1, inp_2)
    
    # setup out
    opl_1 = opl.vecd(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecd(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    out = opl.dot(opl_1, opl_2)
    
    # test
    assert out == pytest.approx(ref)

def test_dot_c32():
    # setup ref
    factor = 5.0
    size = 10
    inp_1 = np.random.randn(size)
    inp_2 = np.random.randn(size)
    ref = np.inner(inp_1, inp_2)
    
    # setup out
    opl_1 = opl.vecx(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecx(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    print(opl_1)
    out = opl.dot(opl_1, opl_2)
    print(out)
    
    # test
    assert out == pytest.approx(ref)

def test_dot_c64():
    # setup ref
    factor = 5.0
    size = 10
    inp_1 = np.random.randn(size) + 1j * np.random.randn(size)
    inp_2 = np.random.randn(size) + 1j * np.random.randn(size)
    ref = np.inner(inp_1, inp_2)
    
    # setup out
    opl_1 = opl.vecz(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecz(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    out = opl.dot(opl_1, opl_2)
    print(out)
    
    # test
    assert out == pytest.approx(ref)

# ===================================================================
# swap
def test_swap_f32():
    # setup ref
    size = 10
    inp_1 = np.random.randn(size)
    inp_2 = np.random.randn(size)
    
    # setup out
    opl_1 = opl.vecf(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecf(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    opl.swap(opl_1, opl_2)
    
    # test
    for i in range(size):
        assert opl_1[i] == pytest.approx(inp_2[i])
    for i in range(size):
        assert opl_2[i] == pytest.approx(inp_1[i])

def test_swap_f64():
    # setup ref
    size = 10
    inp_1 = np.random.randn(size)
    inp_2 = np.random.randn(size)
    
    # setup out
    opl_1 = opl.vecd(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecd(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    out = opl.swap(opl_1, opl_2)
    
    # test
    for i in range(size):
        assert opl_1[i] == pytest.approx(inp_2[i])
    for i in range(size):
        assert opl_2[i] == pytest.approx(inp_1[i])

def test_swap_c32():
    # setup ref
    size = 10
    inp_1 = np.random.randn(size) + 1j * np.random.randn(size)
    inp_2 = np.random.randn(size) + 1j * np.random.randn(size)
    
    # setup out
    opl_1 = opl.vecx(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecx(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    opl.swap(opl_1, opl_2)
    
    # test
    for i in range(size):
        assert opl_1[i] == pytest.approx(inp_2[i])
    for i in range(size):
        assert opl_2[i] == pytest.approx(inp_1[i])

def test_swap_c64():
    # setup ref
    size = 10
    inp_1 = np.random.randn(size) + 1j * np.random.randn(size)
    inp_2 = np.random.randn(size) + 1j * np.random.randn(size)
    
    # setup out
    opl_1 = opl.vecz(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecz(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    opl.swap(opl_1, opl_2)
    
    # test
    for i in range(size):
        assert opl_1[i] == pytest.approx(inp_2[i])
    for i in range(size):
        assert opl_2[i] == pytest.approx(inp_1[i])

# ===================================================================
# copy
def test_copy_f32():
    # setup ref
    size = 10
    inp_1 = np.random.randn(size)
    inp_2 = np.random.randn(size)
    
    # setup out
    opl_1 = opl.vecf(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecf(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    opl.copy(opl_1, opl_2)
    
    # test
    for i in range(size):
        assert opl_2[i] == pytest.approx(inp_1[i])

def test_copy_f64():
    # setup ref
    size = 10
    inp_1 = np.random.randn(size)
    inp_2 = np.random.randn(size)
    
    # setup out
    opl_1 = opl.vecd(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecd(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    out = opl.copy(opl_1, opl_2)
    
    # test
    for i in range(size):
        assert opl_2[i] == pytest.approx(inp_1[i])

def test_copy_c32():
    # setup ref
    size = 10
    inp_1 = np.random.randn(size) + 1j * np.random.randn(size)
    inp_2 = np.random.randn(size) + 1j * np.random.randn(size)
    
    # setup out
    opl_1 = opl.vecx(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecx(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    opl.copy(opl_1, opl_2)
    
    # test
    for i in range(size):
        assert opl_2[i] == pytest.approx(inp_1[i])

def test_copy_c64():
    # setup ref
    size = 10
    inp_1 = np.random.randn(size) + 1j * np.random.randn(size)
    inp_2 = np.random.randn(size) + 1j * np.random.randn(size)
    
    # setup out
    opl_1 = opl.vecz(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecz(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    opl.copy(opl_1, opl_2)
    
    # test
    for i in range(size):
        assert opl_2[i] == pytest.approx(inp_1[i])

# ===================================================================
# axpy
def test_axpy_f32():
    # setup ref
    factor = 5.0
    size = 10
    inp_1 = np.random.randn(size)
    inp_2 = np.random.randn(size)
    ref = factor * inp_1 + inp_2
    
    # setup out
    opl_1 = opl.vecf(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecf(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    opl.axpy(factor, opl_1, opl_2)
    
    # test
    for i in range(size):
        assert opl_2[i] == pytest.approx(ref[i])

def test_axpy_f64():
    # setup ref
    factor = 5.0
    size = 10
    inp_1 = np.random.randn(size)
    inp_2 = np.random.randn(size)
    ref = factor * inp_1 + inp_2
    
    # setup out
    opl_1 = opl.vecd(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecd(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    out = opl.axpy(factor, opl_1, opl_2)
    
    # test
    for i in range(size):
        assert opl_2[i] == pytest.approx(ref[i])

def test_axpy_c32():
    # setup ref
    factor = 5.0
    size = 10
    inp_1 = np.random.randn(size) + 1j * np.random.randn(size)
    inp_2 = np.random.randn(size) + 1j * np.random.randn(size)
    ref = factor * inp_1 + inp_2
    
    # setup out
    opl_1 = opl.vecx(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecx(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    opl.axpy(factor, opl_1, opl_2)
    
    # test
    for i in range(size):
        assert opl_2[i] == pytest.approx(ref[i])

def test_axpy_c64():
    # setup ref
    factor = 5.0
    size = 10
    inp_1 = np.random.randn(size) + 1j * np.random.randn(size)
    inp_2 = np.random.randn(size) + 1j * np.random.randn(size)
    ref = factor * inp_1 + inp_2
    
    # setup out
    opl_1 = opl.vecz(size)
    for i in range(size):
        opl_1[i] = inp_1[i]
    opl_2 = opl.vecz(size)
    for i in range(size):
        opl_2[i] = inp_2[i]
    opl.axpy(factor, opl_1, opl_2)
    
    # test
    for i in range(size):
        assert opl_2[i] == pytest.approx(ref[i])
