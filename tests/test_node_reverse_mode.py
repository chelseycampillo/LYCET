#!/usr/bin/env python3
#File: test_LYCET_operations.py
#Description: test reverse mode evaluation using node class from reverse.py

import pytest
import numpy as np
import LYCET_package.ReverseMode as rm
from LYCET_package.Node import Node
import LYCET_package.LYCET_Operations_Reverse as rmo

"""

Tests for the node class and it's overloaded methods

"""

def test_init_fail():
    """testing for the input of th enode class, shouldn't take a string"""
    with pytest.raises(AssertionError):
        Node('str_2',('str_1',4))

def test_add():
    """testing the addition operator of the node class"""
    a = Node(6,0)
    b = Node(4,0)
    c = a + b
    assert (a.value + b.value == 10) and (c.deriv[0][0] == a) and (c.deriv[0][1] == 1) and (c.deriv[1][0] == b) and (c.deriv[1][1] == 1)

def test_add_2():
    """testing the addition operator of the node class"""
    a = Node(6,0)
    b = 4
    c = a + b
    assert (c.value  == 10) and (c.deriv[0][1] == 1)

def test_radd():
    """testing the addition operator of the node class"""
    a = Node(6,0)
    b = 4
    c = a + b
    d = b + a
    assert (c.value  ==  d.value)

def test_mul():
    """testing the multiplication operator of the node class"""
    a = Node(6,0)
    b = Node(4,0)
    c = a * b
    assert (a.value * b.value == 24) and (c.deriv[0][0] == a) and (c.deriv[0][1] == 4) and (c.deriv[1][0] == b) and (c.deriv[1][1] == 6)

def test_sub():
    """testing the subtraction operator of the node class"""
    a = Node(6)
    b = Node(4)
    c = a - b
    assert (a.value - b.value == 2) and (c.deriv[0][0] == a) and (c.deriv[0][1] == 1) and (c.deriv[1][0] == b) and (c.deriv[1][1] == -1)

def test_sub_2():
    """testing the addition operator of the node class"""
    a = Node(6,0)
    b = 4
    c = a - b
    assert (c.value  == 2) and (c.deriv[0][1] == 1)

def test_truediv():
    """testing the division operator of the node class"""
    a = Node(6,0)
    b = Node(4,0)
    c = a / b
    assert (a.value / b.value == 1.5) and (c.deriv[0][0] == a) and (c.deriv[0][1] == 0.25) and (c.deriv[1][0] == b) and (c.deriv[1][1] == -0.375)

def test_truediv_2():
    """testing the division operator of the node class"""
    a = Node(6,0)
    b = 2
    c = a / b
    assert (c.value == 3)

def test_pow():
    """testing the power operator of the node class"""
    a = Node(6,0)
    b = Node(4,0)
    c = a ** b
    assert (a.value ** b.value == 1296) and (c.deriv[0][0] == a) and (c.deriv[0][1] == 864)

def test_eq():
    """testing the equal operator of the node class."""
    a = Node(5,0)
    b = Node(5,0)
    c = a**b
    d = a**b
    assert a == b
    assert (c.value == d.value) and (c.deriv[0][0] == d.deriv[0][0]) and (c.deriv[0][1] == d.deriv[0][1])

def test_eq_2():
    """testing the equal operator of the node class."""
    a = Node(5,0)
    b = 5
    assert a == b

def test_ne():
    """testing the not equal operator of the node class."""
    a = Node(4,0)
    b = Node(5,0)
    c = a/b
    d = Node(8,0)
    e = Node(4,0)
    f = d/e
    assert a != b
    assert c.value != f.value \
           and a != d \
           and (1/b.value) != (1/e.value) \
           and b != e \
           and (-1*a.value)/(b.value**2) != (-1*d.value)/(e.value**2)

def test_ne_2():
    """testing the not equal operator of the node class."""
    a = Node(4,0)
    b = 5
    assert a != b

def test_ne_3():
    """testing the not equal operator of the node class."""
    a = Node(4,3)
    b = Node(6,5)
    assert a.value != b.value
    assert a.deriv != b.deriv

def test_lt():
    """testing the less than operator of the node class."""
    a = Node(4,0)
    b = Node(5,0)
    c = a*b
    d = Node(6,0)
    e = Node(7,0)
    f = d*e
    assert np.less(c.value, f.value) \
           and np.less(a, d) \
           and np.less(b.value, e.value) \
           and np.less(b, e) \
           and np.less(a.value, d.value)

def test_lt_2():
    """testing the less than operator of the node class."""
    a = Node(4,0)
    b = 7
    assert a < b

def test_lt_3():
    """testing the less than operator of the node class."""
    a = Node(4,0)
    b = Node(5,6)
    assert a.value < b.value
    assert a.deriv < b.deriv

def test_le():
    """testing the less than or equal operator of the node class."""
    a = Node(4,0)
    b = Node(5,0)
    c = b-a
    d = Node(6,0)
    e = Node(8,0)
    f = e-d
    assert np.less_equal(c.value,f.value)\
           and np.less_equal(b, e) \
           and c.deriv[0][1]==1 and f.deriv[0][1] ==1\
           and np.less_equal(a, d) \
           and c.deriv[1][1]==-1 and f.deriv[1][1] ==-1\

def test_le_2():
    """testing the less than or equal operator of the node class."""
    a = Node(4,0)
    b = 5
    assert a < b

def test_le_3():
    """testing the less than or equal operator of the node class."""
    a = Node(4)
    b = Node(5)
    assert np.less_equal(a.value , b.value)

def test_gt():
    """testing the greater than operator of the node class."""
    a = Node(4,0)
    b = Node(5,0)
    c = a+b
    d = Node(6,0)
    e = Node(8,0)
    f = d+e
    assert np.greater(f.value,c.value)\
           and np.greater(d, a) \
           and c.deriv[0][1]==1 and f.deriv[0][1] ==1\
           and np.greater(e, b) \
           and c.deriv[1][1]== 1 and f.deriv[1][1] ==1\

def test_gt_2():
    """testing the greater than operator of the node class."""
    a = Node(4,0)
    b = 5
    assert np.greater(b,a)

def test_gt_2():
    """testing the greater than operator of the node class."""
    a = Node(4,3)
    b = Node(5,6)
    assert np.greater(b.value,a.value)
    assert np.greater(b.deriv,a.deriv)

def test_ge():
    """testing the greater than or equal operator of the node class."""
    a = Node(4,0)
    b = Node(5,0)
    c = a+b
    d = Node(6,0)
    e = Node(8,0)
    f = d+e
    assert np.greater_equal(f.value,c.value)\
           and np.greater_equal(d, a) \
           and c.deriv[0][1]==1 and f.deriv[0][1] ==1\
           and np.greater_equal(e, b) \
           and c.deriv[1][1]== 1 and f.deriv[1][1] ==1\

def test_ge_2():
    """testing the greater than or equal operator of the node class."""
    a = Node(4,0)
    b = 5
    assert np.greater_equal(b,a)

def test_ge_3():
    """testing the greater than or equal operator of the node class."""
    a = Node(4,3)
    b = Node(5,3)
    assert np.greater_equal(b.value,a.value)
    assert np.greater_equal(b.deriv,a.deriv)

"""

test elementary functions

"""

def test_cos():
    assert isinstance(rmo.cos(5), (Node))

def test_tan():
    assert isinstance(rmo.tan(5), (Node))

def test_tan_2():
    with pytest.raises(ValueError):
        rmo.tan(np.pi/2)

def test_exp():
    assert isinstance(rmo.exp(5), (Node))

def test_ln():
    assert isinstance(rmo.ln(5), (Node))

def test_ln2():
    with pytest.raises(ValueError):
        rmo.ln(0)

def test_log():
    x = Node(5)
    base = 5
    y = rmo.log(x,base)
    assert y.value == np.log(5)/np.log(5)

def test_log():
    assert isinstance(rmo.log(5,3), (Node))

def test_arcsin():
    assert isinstance(rmo.arcsin(0.5), (Node))

def test_arcsin2():
    with pytest.raises(ValueError):
        rmo.arcsin(6)

def test_arccos():
    assert isinstance(rmo.arccos(0.5), (Node))

def test_arccos2():
    with pytest.raises(ValueError):
        rmo.arccos(6)

def test_arctan():
    assert isinstance(rmo.arctan(5), (Node))

def test_sinh():
    assert isinstance(rmo.sinh(5), (Node))

def test_ccsh():
    assert isinstance(rmo.cosh(5), (Node))

def test_tanh():
    assert isinstance(rmo.tanh(5), (Node))

def test_sigmoid():
    assert isinstance(rmo.sigmoid(5), (Node))

"""

Tests that reverse mode works with different elementary functions

"""


def test_elementary_operations():
    """ test the value part of the node"""
    # create node to pass to a function
    X1 = Node(0.5)
    #define reverse function log for the test since it takes a second argument
    def rmolog(x1):
        val = np.log(x1.value) / np.log(5)
        deriv = (
            (x1, 1 / (x1.value * np.log(5))),
        )
        return Node(val, list(deriv))
    # add all reverse mode functions to list
    rm_functions = [rmo.sin,
    rmo.cos,
    rmo.tan,
    rmo.exp,
    rmo.ln,
    rmolog,
    rmo.arcsin,
    rmo.arccos,
    rmo.arctan,
    rmo.sinh,
    rmo.cosh,
    rmo.tanh,
    rmo.sigmoid]
    # empty list of results
    rm_output_value = []
    # loop through reverse functions get the value part append to output value
    for func in rm_functions:
        rm_output_value.append(func(X1).value)
    # define variable to evaluate function
    x1 = 0.5
    # define sigmoid function since numpy doesn't have an equivalent
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # define the log function with base 5
    def nplog(x):
        return np.log(x1) / np.log(5)
    # store the np functions we will test
    np_functions = [np.sin,
    np.cos,
    np.tan,
    np.exp,
    np.log,
    nplog,
    np.arcsin,
    np.arccos,
    np.arctan,
    np.sinh,
    np.cosh,
    np.tanh,
    sigmoid]
    # empty list of values
    np_output_value = []
    # loop through functions and evaluate them at x1
    for func in np_functions:
        np_output_value.append(func(x1))
    # assert that np values equals rv values
    assert rm_output_value == np_output_value

    """ test the first element of the deriv part of the node"""
    def sin_der(x1):
        return np.cos(x1)
    def cos_der(x1):
        return -np.sin(x1)
    def tan_der(x1):
        return 1 / ((np.cos(x1)) ** 2)
    def exp_der(x1):
        return np.exp(x1)
    def ln_der(x1):
        return 1 / x1
    def log_der(x1):
        return 1 / (x1 * np.log(5))
    def arcsin_Der(x1):
        return 1 / np.sqrt(1 - x1 ** 2)
    def arccos_der(x1):
        return -1 / np.sqrt(1 - x1 ** 2)
    def arctan_der(x1):
        return 1 / ((x1 ** 2) + 1)
    def sinh_der(x1):
        return np.cosh(x1)
    def cosh_der(x1):
        return np.sinh(x1)
    def tanh_der(x1):
        return 1 - np.tanh(x1) ** 2
    def sigmoid_der(x1):
        return (1 / (1 + np.exp(-x1))) * (1 - (1 / (1 + np.exp(-x1))))

    rm_output_deriv = []

    for func in rm_functions:
        assert func(X1).deriv[0][0] == X1
        rm_output_deriv.append(func(X1).deriv[0][1])
    np_functions_deriv = [sin_der,
        cos_der,
        tan_der,
        exp_der,
        ln_der,
        log_der,
        arcsin_Der,
        arccos_der,
        arctan_der,
        sinh_der,
        cosh_der,
        tanh_der,
        sigmoid_der]
    np_output_deriv = []
    for func in np_functions_deriv:
        np_output_deriv.append(func(x1))
    assert np_output_deriv == rm_output_deriv

def test_reverse_mode_1():
    # test the reverse mode method for function nd.cos(x1 + x2) + (x3 * x2 ** 3), return function evaluated at x and derivative
    f = lambda x1, x2, x3: rmo.cos(x1 + x2) + (x3 * x2 ** 3)
    x = [1, 2, 3]
    eval_func, eval_deriv = rm.ReverseMode(f, x)

    assert eval_func == np.cos(1+2) + (3*2**3)
    assert eval_deriv[0] == -1*np.sin(1+2)
    assert eval_deriv[1] == (-1*np.sin(1+2)+3*3*(2**2))
    assert eval_deriv[2] == 2**3

def test_reverse_mode_2():
    # test the reverse mode method for function nd.sin(x1)*nd.cos(x1), return function evaluated at x and derivative
    f = lambda x1: rmo.sin(x1)*rmo.cos(x1)
    x = 1
    eval_func, eval_deriv = rm.ReverseMode(f, x)

    assert eval_func == np.sin(1)*np.cos(1)
    assert (np.abs(eval_deriv-np.cos(2*1))<np.finfo(float).eps)

def test_reverse_mode_3():
    # test the reverse mode method for function nd.sin(x1)*x1, return function evaluated at x and derivative
    f = lambda x1: rmo.sin(5)*x1
    x = 5
    eval_func, eval_deriv = rm.ReverseMode(f, x)

    assert eval_func == np.sin(5)*5
    assert eval_deriv == np.sin(5)

def test_reverse_mode_4():
    # test the reverse mode method for function x1*rmo.sin(5) + x2*10, return function evaluated at x and derivative
    f = lambda x1, x2: x1*rmo.sin(5) + x2*10
    x = [5,10]
    eval_func, eval_deriv = rm.ReverseMode(f, x)

    assert eval_func == np.sin(5)*5 + 10*10
    assert eval_deriv[0] == np.sin(5)
    assert eval_deriv[1] == 10

def test_reverse_mode_5():
    # test the reverse mode method for function 5*x1 + 10*x2, return function evaluated at x and derivative
    f = lambda x1, x2: 5*x1 + 10*x2
    x = [5,10]
    eval_func, eval_deriv = rm.ReverseMode(f, x)

    assert eval_func == 125
    assert eval_deriv[0] == 5
    assert eval_deriv[1] == 10

def test_reverse_mode_6():
    # test the reverse mode method for function 10*x2 + 5*x1, return function evaluated at x and derivative
    f = lambda x1, x2: 10*x2 + 5*x1
    x = [5,10]
    eval_func, eval_deriv = rm.ReverseMode(f, x)

    assert eval_func == 125
    assert eval_deriv[0] == 5
    assert eval_deriv[1] == 10

def test_reverse_mode_7():
    # test the reverse mode method for function nd.ln(x1/x2), return function evaluated at x and derivative
    f = lambda x1, x2: rmo.ln(x1/x2)
    x = [10, 50]
    eval_func, eval_deriv = rm.ReverseMode(f, x)

    assert eval_func == np.log(10/50)
    assert eval_deriv[0] == 1/10
    assert eval_deriv[1] == -1*(1/50)

def test_reverse_mode_8():
    # test the reverse mode method for function nd.cos(x1) + nd.tanh(x2) + (x3 / x4) + (x5**(nd.cosh(x6))) + (x7 - x1), return function evaluated at x and derivative
    f = lambda x1, x2, x3 , x4, x5 , x6 ,x7: rmo.cos(x1) + rmo.tanh(x2) + (x3 / x4) + (x5**(rmo.cosh(x6))) + (x7 - x1)
    x = [0.5912, 0.3242, 0.8177, 2.9087, 5.3690, 6.4394, 3.1917]
    eval_func, eval_deriv = rm.ReverseMode(f, x)

    assert eval_func == np.cos(0.5912) + np.tanh(0.3242) + (0.8177 / 2.9087) + (5.3690 ** (np.cosh(6.4394))) + (3.1917 - 0.5912)
    assert eval_deriv[0] == -1*np.sin(0.5912) -1
    assert eval_deriv[1] == (2/(np.exp(0.3242)+ np.exp(-0.3242)))**2
    assert eval_deriv[2] == 1/2.9087
    assert eval_deriv[3] == -1*0.8177/(2.9087**2)
    assert eval_deriv[4] == (5.3690**(np.cosh(6.4394)-1))*np.cosh(6.4394)
    assert eval_deriv[5] == 0
    assert eval_deriv[6] == 1

if __name__ == '__main__':
    test_init_fail()
    test_add_2()
    test_mul()
    test_sub()
    test_truediv_2()
    test_pow()
    test_eq_2()
    test_ne_2()
    test_lt_2()
    test_le_3()
    test_gt_2()
    test_ge()
    test_elementary_operations()
    test_reverse_mode_1
    test_reverse_mode_2
    test_reverse_mode_3
    test_reverse_mode_4
    test_reverse_mode_5
    test_reverse_mode_6
    test_reverse_mode_7
    test_reverse_mode_8
    test_add_2()
    test_radd()
    test_sub_2()
    test_truediv_2()
    test_eq_2()
    test_ne_2()
    test_ne_3()
    test_lt_2()
    test_lt_3()
    test_le_2()
    test_le_3()
    test_gt_2()
    test_gt_2()
    test_ge_2()
    test_ge_3()
    test_cos()
    test_tan()
    test_tan_2()
    test_exp()
    test_ln()
    test_ln2()
    test_log()
    test_log()
    test_arcsin()
    test_arcsin2()
    test_arccos()
    test_arccos2()
    test_arctan()
    test_sinh()
    test_ccsh()
    test_tanh()
    test_sigmoid()