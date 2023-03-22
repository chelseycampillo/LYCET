#!/usr/bin/env python3
#File: test_LYCET_operations.py
#Description: test forward mode evaluation using DualNumber and LYCET_operations classes

import pytest
import numpy as np
import LYCET_package.LYCET_Operations_Forward as lycet
from LYCET_package.DualNumber import DualNumber

def test_sin():
    #test lycet forward mode elementary function sin
    inputs = [DualNumber(5, 1), DualNumber(5.1, 1), 5, 5.1]
    for i in inputs:
        output = lycet.sin(i)
        if isinstance(i, DualNumber):
            assert (output.real) == np.sin(i.real) and (output.dual == np.cos(i.real))
        else:
            assert (output) == np.sin(i)

def test_cos():
    # test lycet forward mode elementary function cos
    inputs = [DualNumber(5, 1), DualNumber(5.1, 1),5,5.1]
    for i in inputs:
        output = lycet.cos(i)
        if isinstance(i, DualNumber):
            assert (output.real) == np.cos(i.real) and (output.dual == -np.sin(i.real))
        else:
            assert (output) == np.cos(i)

def test_tan():
    # test lycet forward mode elementary function tan
    inputs = [DualNumber(5, 1), DualNumber(5.1, 1),5,5.1]
    for i in inputs:
        output = lycet.tan(i)
        if isinstance(i, DualNumber):
            assert (output.real) == np.tan(i.real) and (output.dual == (1 + (np.sin(i.real)**2)/(np.cos(i.real)**2)))
        else:
            assert (output) == np.tan(i)

def test_ln():
    # test lycet forward mode elementary function ln
    inputs = [DualNumber(5, 1), DualNumber(5.1, 1),5,5.1]
    for i in inputs:
        output = lycet.ln(i)
        if isinstance(i, DualNumber):
            assert (output.real) == np.log(i.real) and (output.dual == 1/i.real)
        else:
            assert (output) == np.log(i)

def test_ln_inval_dom():
    # test lycet forward mode elementary function ln for invalid domain inputs
    inputs = [DualNumber(-5, 1), DualNumber(-5.1, 1),DualNumber(0, 1)]
    for i in inputs:
        if isinstance(i, DualNumber):
            with pytest.raises(ValueError):
                lycet.ln(i)

def test_log():
    # test lycet forward mode elementary function log
    inputs = [DualNumber(5, 1), DualNumber(5.1, 1),5,5.1]
    base = [3,3.1]
    for i in inputs:
        for x in base:
            output = lycet.log(i,x)
            if isinstance(i, DualNumber):
                assert np.abs((output.real) - np.log(i.real)/np.log(x)) < np.finfo(float).eps and np.abs((output.dual - 1/(i.real*np.log(x)))) < np.finfo(float).eps
            else:
                assert np.abs((output) - np.log(i)/np.log(x)) < np.finfo(float).eps

def test_log_inval_dom():
    # test lycet forward mode elementary function log for invalid domain inputs
    inputs = [DualNumber(-5, 1), DualNumber(-5.1, 1),DualNumber(0, 1),0,-5,-5.1]
    base = [-3,-3.1]
    for i in inputs:
        for x in base:
            if isinstance(i, DualNumber):
                with pytest.raises(ValueError):
                    lycet.log(i.real,x)
            else:
                with pytest.raises(ValueError):
                    lycet.log(i,x)

def test_exp():
    # test lycet forward mode elementary function exp
    inputs = [DualNumber(5, 1), DualNumber(5.1, 1),5,5.1]
    for i in inputs:
        output = lycet.exp(i)
        if isinstance(i, DualNumber):
            assert (output.real) == np.exp(i.real) and output.dual == np.exp(i.real)
        else:
            assert (output) == np.exp(i)

def test_arcsin():
    # test lycet forward mode elementary function arcsin
    inputs = [DualNumber(0.5, 1), 0.5]
    for i in inputs:
        output = lycet.arcsin(i)
        if isinstance(i, DualNumber):
            assert (output.real) == np.arcsin(i.real) and output.dual == 1 / np.sqrt(1 - i.real ** 2)
        else:
            assert (output.real) == np.arcsin(i)

def test_arcsin_inval_dom():
    # test lycet forward mode elementary function arcsin for invliad domain input
    inputs = [DualNumber(5,1), DualNumber(5.1,1)]
    for i in inputs:
        with pytest.raises(ValueError):
                lycet.arcsin(i)

def test_arccos():
    # test lycet forward mode elementary function arcos
    inputs = [DualNumber(0.5, 1),0.5]
    for i in inputs:
        output = lycet.arccos(i)
        if isinstance(i, DualNumber):
            assert (output.real) == np.arccos(i.real) and output.dual == -1 / np.sqrt(1 - i.real ** 2)
        else:
            assert (output) == np.arccos(i)

def test_arccos_inval_dom():
    # test lycet forward mode elementary function arcos for inval domain
    inputs = [DualNumber(5,1), DualNumber(5.1,1)]
    for i in inputs:
        with pytest.raises(ValueError):
            lycet.arccos(i)

def test_arctan():
    # test lycet forward mode elementary function arctam
    inputs = [DualNumber(5,1), DualNumber(5.1,1),5,5.1]
    for i in inputs:
        output = lycet.arctan(i)
        if isinstance(i, DualNumber):
            assert (output.real) == np.arctan(i.real) and output.dual == 1/((i.real ** 2)+1)
        else:
            assert (output) == np.arctan(i)

def test_sinh():
    # test lycet forward mode elementary function sinh
    inputs = [DualNumber(5,1), DualNumber(5.1,1),5,5.1]
    for i in inputs:
        output = lycet.sinh(i)
        if isinstance(i, DualNumber):
            assert np.abs(output.real - np.sinh(i.real)) < np.finfo(float).eps*1000 and np.abs(output.dual - np.cosh(i.real)) < np.finfo(float).eps
        else:
            assert np.abs(output - np.sinh(i)) < np.finfo(float).eps*1000

def test_cosh():
    # test lycet forward mode elementary function cosh
    inputs = [DualNumber(5,1), DualNumber(5.1,1),5,5.1]
    for i in inputs:
        output = lycet.cosh(i)
        if isinstance(i, DualNumber):
            assert output.real == np.cosh(i.real) and output.dual == (np.exp(i.real) - np.exp(-i.real))/2
        else:
            assert output == np.cosh(i)

def test_tanh():
    # test lycet forward mode elementary function tanh
    inputs = [DualNumber(5,1), DualNumber(5.1,1),5,5.1]
    for i in inputs:
        output = lycet.tanh(i)
        if isinstance(i, DualNumber):
            assert (output.real) - np.tanh(i.real) and output.dual - (np.exp(i.real) - np.exp(-i.real))/(np.exp(i.real) + np.exp(-i.real))
        else:
            assert (output) == np.tanh(i)

def test_sigmoid():
    # test lycet forward mode elementary function sigmoid
    inputs = [DualNumber(5,1), DualNumber(5.1,1),5,5.1]
    for i in inputs:
        output = lycet.sigmoid(i)
        if isinstance(i, DualNumber):
            assert (output.real) == 1/(1 + np.exp(-i.real)) and output.dual == (np.exp(-i.real)/(1+np.exp(-i.real))**2)
        else:
            assert (output) == 1/(1 + np.exp(-i))

if __name__ == '__main__':
    test_sin()
    test_cos()
    test_tan()
    test_ln()
    test_ln_inval_dom()
    test_log()
    test_log_inval_dom()
    test_exp()
    test_arcsin()
    test_arcsin_inval_dom()
    test_arccos()
    test_arccos_inval_dom()
    test_arctan()
    test_sinh()
    test_cosh()
    test_tanh()
    test_sigmoid()