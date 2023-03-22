import pytest
import math
import numpy as np
import LYCET_package.ForwardMode as fm
import LYCET_package.LYCET_Operations_Forward as lycet
from LYCET_package.DualNumber import DualNumber

def test_R_R():
    # R -> R
    # y = x^2
    # dy/dx = 2 x
    for i in [1, 1.0]:
        output = fm.ForwardMode(f=lambda x: x**2, x=3*i)
        assert output == (3**2, 2*3)
        output = fm.ForwardMode(f=lambda x: x**2, x=3*i, gradient=True)
        assert output == 2*3
        output = fm.ForwardMode(f=lambda x: x**2, x=3*i, jacobian=True)
        assert output == 2*3

def test_R_R1():
    # R -> R^1
    for i in [1, 1.0]:
        output = fm.ForwardMode(f=lambda x: (x**2,), x=3*i, jacobian=True)
        assert output == (2*3,)

def test_R1_R():
    # R^1 -> R
    for i in [1, 1.0]:
        output = fm.ForwardMode(f=lambda x: x[0]**2, x=(3*i,), gradient=True)
        assert output == 2*3
        output = fm.ForwardMode(f=lambda x: x[0]**2, x=(3*i,), jacobian=True)
        assert output == 2*3

def test_R1_R1():
    # R^1 -> R^1
    for i in [1, 1.0]:
        output = fm.ForwardMode(f=lambda x: (x[0]**2,), x=(3*i,), jacobian=True)
        assert output == (2*3,)

def test_Rn_R():
    # R^n -> R
    # y = x0^2 + x1^3 + 5 x0 x1
    # dy/dx0 = 2 x0 + 5 x1
    # dy/dx1 = 3 x1^2 + 5 x0
    for i in [(1, 1), (1, 1.0), (1.0, 1), (1.0, 1.0)]:
        output = fm.ForwardMode(f=lambda x: x[0]**2 + x[1]**3 + 5*x[0]*x[1], x=(7*i[0], 11*i[1]), gradient=True)
        assert np.array_equal(output, (2 * 7 + 5 * 11, 3 * 11**2 + 5 * 7))
        output = fm.ForwardMode(f=lambda x: x[0]**2 + x[1]**3 + 5*x[0]*x[1], x=(7*i[0], 11*i[1]), jacobian=True)
        assert np.array_equal(output, (2 * 7 + 5 * 11, 3 * 11**2 + 5 * 7))

def test_Rn_R1():
    # R^n -> R^1
    for i in [(1, 1), (1, 1.0), (1.0, 1), (1.0, 1.0)]:
        output = fm.ForwardMode(f=lambda x: (x[0]**2 + x[1]**3 + 5*x[0]*x[1],), x=(7*i[0], 11*i[1]), jacobian=True)
        assert np.array_equal(output, ((2 * 7 + 5 * 11, 3 * 11**2 + 5 * 7),))

def test_R_Rn():
    # R -> R^n
    # y = (x^2, x^3)
    # dy/dx = (2 x, 3 x^2)
    for i in [1, 1.0]:
        output = fm.ForwardMode(f=lambda x: (x**2, x**3), x=5*i, jacobian=True)
        assert np.array_equal(output, ((2 * 5,), (3 * 5**2,)))

def test_R1_Rn():
    # R^1 -> R^n
    for i in [1, 1.0]:
        output = fm.ForwardMode(f=lambda x: (x[0]**2, x[0]**3), x=(5*i,), jacobian=True)
        assert np.array_equal(output, ((2 * 5,), (3 * 5**2,)))

def test_Rn_Rm():
    # R^n -> R^m
    # y = (x0^2 + x1^3 + 5 x0 x1, x0^7 + x1^11 + 13 x0 x1)
    # dy/dx0 = (2 x0 + 5 x1, 7 x0^6 + 13 x1)
    # dy/dx1 = (3 x1^2 + 5 x0, 11 x1^10 + 13 x0)
    for i in [(1, 1), (1, 1.0), (1.0, 1), (1.0, 1.0)]:
        x0 = 17*i[0]
        x1 = 19*i[1]
        y = (x0**2 + x1**3 + 5 * x0 * x1, x0**7 + x1**11 + 13 * x0 * x1)
        dydx0 = (2 * x0 + 5 * x1, 7 * x0**6 + 13 * x1)
        dydx1 = (3 * x1**2 + 5 * x0, 11 * x1**10 + 13 * x0)
        output = fm.ForwardMode(f=lambda x: (x[0]**2 + x[1]**3 + 5 * x[0] * x[1], x[0]**7 + x[1]**11 + 13 * x[0] * x[1]), x=(x0, x1), jacobian=True)
        assert np.array_equal(output,
            ((dydx0[0], dydx1[0]),
            (dydx0[1], dydx1[1])))
        output = fm.ForwardMode(f=lambda x: (x[0]**2 + x[1]**3 + 5 * x[0] * x[1], x[0]**7 + x[1]**11 + 13 * x[0] * x[1]), x=(x0, x1), p=[1,0])
        assert np.array_equal(output, ((y[0], y[1]), (dydx0[0], dydx0[1])))
        output = fm.ForwardMode(f=lambda x: (x[0]**2 + x[1]**3 + 5 * x[0] * x[1], x[0]**7 + x[1]**11 + 13 * x[0] * x[1]), x=(x0, x1), p=[0,1])
        assert np.array_equal(output, ((y[0], y[1]), (dydx1[0], dydx1[1])))
        output = fm.ForwardMode(f=lambda x: (x[0]**2 + x[1]**3 + 5 * x[0] * x[1], x[0]**7 + x[1]**11 + 13 * x[0] * x[1]), x=(x0, x1), p=[1,1])
        assert np.array_equal(output, ((y[0], y[1]), (dydx0[0] + dydx1[0], dydx0[1] + dydx1[1])))

if __name__ == '__main__':
    test_R_R()
    test_R_R1()
    test_R1_R()
    test_R1_R1()
    test_Rn_R()
    test_Rn_R1()
    test_R_Rn()
    test_R1_Rn()
    test_Rn_Rm()
