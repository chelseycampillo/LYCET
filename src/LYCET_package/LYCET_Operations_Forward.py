#!/usr/bin/env python3
# File: LYCET_Operations_Forward.py
# Description: Define functions (which do not have a magic function) and their derivatives for Forward Mode

from .DualNumber import DualNumber
import numpy as np

def sin(z):
    """
    Overloaded elementary trig function sine

    Parameters
    =======
    z: must be DualNumber, int, or float 

    Returns
    =======
    A new DualNumber object with the 
    sine computation done on the real (function) and dual (derivative)

    EXAMPLES
    =======
    >>> x = DualNumber(2, 3)
    >>> f1 = fm.sin(x)
    >>> print(f1)
    Dual Number (real=0.9092974268256817, dual=-1.2484405096414273)
    """
    assert isinstance(z, (DualNumber)) or isinstance(z, (np.floating, float)) or np.issubdtype(type(z), np.integer), f"The object {z} is not a Dual Number, integer, or float"
    if np.issubdtype(type(z), np.integer) or isinstance(z, (np.floating, float)):
        return np.sin(z)
    return DualNumber(np.sin(z.real), np.cos(z.real)*z.dual)
    
def cos(z):
    """
    Overloaded elementary trig function cosine

    Parameters
    =======
    z: must be DualNumber, int, or float 

    Returns
    =======
    A new DualNumber object with the 
    cosine computation done on the real (function) and dual (derivative)

    EXAMPLES
    =======
    >>> x = DualNumber(2, 3)
    >>> f1 = fm.cos(x)
    >>> print(f1)
    Dual Number (real=-0.4161468365471424, dual=-2.727892280477045)
    """
    assert isinstance(z, (DualNumber)) or isinstance(z, (np.floating, float)) or np.issubdtype(type(z), np.integer), f"The object {z} is not a Dual Number, integer, or float"
    if np.issubdtype(type(z), np.integer) or isinstance(z, (np.floating, float)):
        return np.cos(z) 
    return DualNumber(np.cos(z.real), -np.sin(z.real)*z.dual) 

def tan(z):
    """
    Overloaded elementary trig function tangent

    Parameters
    =======
    z: must be DualNumber, int, or float 

    Returns
    =======
    A new DualNumber object with the 
    tangent computation done on the real (function) and dual (derivative)

    EXAMPLES
    =======
    >>> x = DualNumber(2, 3)
    >>> f1 = fm.tan(x)
    >>> print(f1)
    Dual Number (real=-2.185039863261519, dual=17.323197612125753)
    """
    assert isinstance(z, (DualNumber)) or isinstance(z, (np.floating, float)) or np.issubdtype(type(z), np.integer), f"The object {z} is not a Dual Number, integer, or float"
    if np.issubdtype(type(z), np.integer) or isinstance(z, (np.floating, float)):
        return np.tan(z)
    if (np.abs(np.cos(z.real)) < np.finfo(float).eps):
        raise ValueError("Invalid domain for Tan.")
    return DualNumber(np.tan(z.real), (1 + (np.sin(z.real)**2)/(np.cos(z.real)**2))*z.dual) 
    

def ln(z):
    """
    Overloaded elementary function natural log

    Parameters
    =======
    z: must be DualNumber, int, or float  

    Returns
    =======
    A new DualNumber object with the natural
    logarithmic computation done on the real (function) and dual (derivative)

    EXAMPLES
    =======
    >>> x = DualNumber(2, 3)
    >>> f1 = fm.ln(x)
    >>> print(f1)
    Dual Number (real=0.6931471805599453, dual=1.5)
    """
    assert isinstance(z, (DualNumber)) or isinstance(z, (np.floating, float)) or np.issubdtype(type(z), np.integer), f"The object {z} is not a Dual Number, integer, or float"
    if np.issubdtype(type(z), np.integer) or isinstance(z, (np.floating, float)):
        return np.log(z)
    if (z.real <= 0):
        raise ValueError("Cannot compute logarithm of negative numbers or 0")
    return DualNumber(np.log(z.real), z.dual/z.real)

def log(z, base):
    """
    Overloaded elementary function logarithmic with base b

    Parameters
    =======
    z: must be DualNumber, int, or float 
    base: must be int or float

    Returns
    =======
    A new DualNumber object with the
    logarithmic computation done on the real (function) and dual (derivative)

    EXAMPLES
    =======
    >>> x = DualNumber(2, 3)
    >>> f1 = fm.log(x, 10)
    >>> print(f1)
    Dual Number (real=0.30102999566398114, dual=0.6514417228548777)
    """
    assert isinstance(z, (DualNumber)) or isinstance(z, (np.floating, float)) or np.issubdtype(type(z), np.integer), f"The object {z} is not a Dual Number, integer, or float"
    if base <= 0:
        raise ValueError("Cannot compute logarithm of negative numbers")
    if np.issubdtype(type(z), np.integer) or isinstance(z, (np.floating, float)):
        return np.log(z)/np.log(base)
    return ln(z)/np.log(base)

def exp(z):
    """
    Overloaded elementary function exponential

    Parameters
    =======
    z: must be DualNumber, int, or float 

    Returns
    =======
    A new DualNumber object with the
    exponential computation done on the real (function) and dual (derivative)

    EXAMPLES
    =======
    >>> x = DualNumber(2, 3)
    >>> f1 = fm.exp(x)
    >>> print(f1)
    Dual Number (real=7.38905609893065, dual=22.16716829679195)
    """
    assert isinstance(z, (DualNumber)) or isinstance(z, (np.floating, float)) or np.issubdtype(type(z), np.integer), f"The object {z} is not a Dual Number, integer, or float"
    if np.issubdtype(type(z), np.integer) or isinstance(z, (np.floating, float)):
        return np.exp(z)
    return DualNumber(np.exp(z.real), z.dual*np.exp(z.real))


def arcsin(z):
    """
    Overloaded elementary function inverse sine

    Parameters
    =======
    z: must be DualNumber, int, or float 

    Returns
    =======
    A new DualNumber object with the
    inverse sine computation done on the real (function) and dual (derivative)

    EXAMPLES
    =======
    >>> x = DualNumber(0.5, 0)
    >>> f1 = fm.arcsin(x)
    >>> print(f1)
    Dual Number (real=0.5235987755982988, dual=0.0)
    """
    assert isinstance(z, (DualNumber)) or isinstance(z, (np.floating, float)) or np.issubdtype(type(z), np.integer), f"The object {z} is not a Dual Number, integer, or float"
    if np.issubdtype(type(z), np.integer) or isinstance(z, (np.floating, float)):
        return np.arcsin(z)
    if -1 > z.real or z.real > 1:
        raise ValueError("Invalid Domain, must between -1 and 1")
    new_arcsin = np.arcsin(z.real)
    der_arcsin = z.dual * 1 / np.sqrt(1 - z.real ** 2)
    return DualNumber(new_arcsin, der_arcsin)

def arccos(z):
    """
    Overloaded elementary function inverse cosine

    Parameters
    =======
    z: must be DualNumber, int, or float 

    Returns
    =======
    A new DualNumber object with the
    inverse cosine computation done on the real (function) and dual (derivative)

    EXAMPLES
    =======
    >>> x = DualNumber(0.5, 0)
    >>> f1 = fm.arccos(x)
    >>> print(f1)
    Dual Number (real=1.0471975511965976, dual=0.0)
    """
    assert isinstance(z, (DualNumber)) or isinstance(z, (np.floating, float)) or np.issubdtype(type(z), np.integer), f"The object {z} is not a Dual Number, integer, or float"
    if np.issubdtype(type(z), np.integer) or isinstance(z, (np.floating, float)):
        return np.arccos(z)
    if -1 > z.real or z.real > 1:
        raise ValueError("Invalid Domain, must between -1 and 1")
    new_arccos = np.arccos(z.real)
    der_arccos = -z.dual * 1 / np.sqrt(1 - z.real ** 2)
    return DualNumber(new_arccos, der_arccos)

def arctan(z):
    """
    Overloaded elementary function inverse tangent

    Parameters
    =======
    z: must be DualNumber, int, or float 

    Returns
    =======
    A new DualNumber object with the
    inverse tangent computation done on the real (function) and dual (derivative)

    EXAMPLES
    =======
    >>> x = DualNumber(2, 3)
    >>> f1 = fm.arctan(x)
    >>> print(f1)
    Dual Number (real=1.1071487177940906, dual=0.6)
    """
    assert isinstance(z, (DualNumber)) or isinstance(z, (np.floating, float)) or np.issubdtype(type(z), np.integer), f"The object {z} is not a Dual Number, integer, or float"
    if np.issubdtype(type(z), np.integer) or isinstance(z, (np.floating, float)):
        return np.arctan(z)
    new_arctan = np.arctan(z.real)
    der_arctan= z.dual * 1 / ((z.real ** 2) + 1)
    return DualNumber(new_arctan, der_arctan)

def sinh(z):
    """
    Overloaded elementary function hyperbolic sine

    Parameters
    =======
    z: must be DualNumber, int, or float 

    Returns
    =======
    A new DualNumber object with the
    hyperbolic sine computation done on the real (function) and dual (derivative)

    EXAMPLES
    =======
    >>> x = DualNumber(2, 3)
    >>> f1 = fm.sinh(x)
    >>> print(f1)
    Dual Number (real=3.626860407847019, dual=11.286587073250894)
    """
    assert isinstance(z, (DualNumber)) or isinstance(z, (np.floating, float)) or np.issubdtype(type(z), np.integer), f"The object {z} is not a Dual Number, integer, or float"
    if np.issubdtype(type(z), np.integer) or isinstance(z, (np.floating, float)):
        return np.sinh(z)
    return (exp(z) - exp(-z))/2

def cosh(z):
    """
    Overloaded elementary function hyperbolic cosine

    Parameters
    =======
    z: must be DualNumber, int, or float 

    Returns
    =======
    A new DualNumber object with the
    hyperbolic cosine computation done on the real (function) and dual (derivative)

    EXAMPLES
    =======
    >>> x = DualNumber(2, 3)
    >>> f1 = fm.cosh(x)
    >>> print(f1)
    Dual Number (real=3.7621956910836314, dual=10.880581223541055)
    """
    assert isinstance(z, (DualNumber)) or isinstance(z, (np.floating, float)) or np.issubdtype(type(z), np.integer), f"The object {z} is not a Dual Number, integer, or float"
    if np.issubdtype(type(z), np.integer) or isinstance(z, (np.floating, float)):
        return np.cosh(z)
    return (exp(z) + exp(-z))/2

def tanh(z):
    """
    Overloaded elementary function hyperbolic tangent

    Parameters
    =======
    z: must be DualNumber, int, or float 

    Returns
    =======
    A new DualNumber object with the
    hyperbolic tangent computation done on the real (function) and dual (derivative)

    EXAMPLES
    =======
    >>> x = DualNumber(2, 3)
    >>> f1 = fm.tanh(x)
    >>> print(f1)
    Dual Number (real=0.964027580075817, dual=0.2119524745594934)
    """
    assert isinstance(z, (DualNumber)) or isinstance(z, (np.floating, float)) or np.issubdtype(type(z), np.integer), f"The object {z} is not a Dual Number, integer, or float"
    if np.issubdtype(type(z), np.integer) or isinstance(z, (np.floating, float)):
        return np.tanh(z)
    return (exp(z) - exp(-z))/(exp(z) + exp(-z))

def sigmoid(z):
    """
    Overloaded elementary function sigmoid

    Parameters
    =======
    z: must be DualNumber, int, or float  

    Returns
    =======
    A new DualNumber object with the
    sigmoid computation done on the real (function) and dual (derivative)

    EXAMPLES
    =======
    >>> x = DualNumber(2, 3)
    >>> f1 = fm.sigmoid(x)
    >>> print(f1)
    Dual Number (real=0.8807970779778823, dual=0.3149807562105195)
    """
    assert isinstance(z, (DualNumber)) or isinstance(z, (np.floating, float)) or np.issubdtype(type(z), np.integer), f"The object {z} is not a Dual Number, integer, or float"
    if np.issubdtype(type(z), np.integer) or isinstance(z, (np.floating, float)):
        return 1/(1 + np.exp(-z))

    return 1/(1 + exp(-z))

