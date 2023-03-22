#!/usr/bin/env python3
#File: LYCET_Operations_Reverse.py
#Description: Define functions (which do not have a magic function) for Reverse Mode

import numpy as np
from .Node import Node

def sin(x):
    """
    Overloaded elementary trig function sine

    Parameters
    =======
    x: must be Node, int, or float 

    Returns
    =======
    A new Node object with the derivative of the
    sine computation done and the evaluation of the gradient

    EXAMPLES
    =======
    >>> x = Node(2)
    >>> f1 = rmo.sin(x)
    >>> print(f1)
    [(Reverse-Mode AD:(f(x)=0.9092974268256817, J=[((f(x)=2, J=()), -0.4161468365471424)]
    """
    assert isinstance(x, (Node, int, float)), f"The object {x} is not a Node, integer, or float"
    if isinstance(x, (int, float)):
        x = Node(x, )
    val = np.sin(x.value)
    deriv = (
        (x, np.cos(x.value)),
    )
    return Node(val, list(deriv))

def cos(x):
    """
    Overloaded elementary trig function cosine

    Parameters
    =======
    x: must be Node, int, or float 

    Returns
    =======
    A new Node object with the derivative of the
    cosine computation done and the evaluation of the gradient

    EXAMPLES
    =======
    >>> x = Node(2)
    >>> f1 = rmo.cos(x)
    >>> f1.value
    -0.4161468365471424
    >>> f1.deriv
    [(Reverse-Mode AD: (f(x)=2, J=()), -0.9092974268256817)]
    """
    assert isinstance(x, (Node, int, float)), f"The object {x} is not a Node, integer, or float"
    if isinstance(x, (int, float)):
        x = Node(x, )
    val = np.cos(x.value)
    deriv = (
        (x, -np.sin(x.value)),
    )
    return Node(val, list(deriv))

def tan(x):
    """
    Overloaded elementary trig function tangent

    Parameters
    =======
    x: must be Node, int, or float 

    Returns
    =======
    A new Node object with the derivative of the
    tangent computation done and the evaluation of the gradient

    EXAMPLES
    =======
    >>> x = Node(2)
    >>> f1 = rmo.tan(x)
    >>> f1.value
    -2.185039863261519
    >>> f1.deriv
    [(Reverse-Mode AD: (f(x)=2, J=()), 5.774399204041917)]
    """
    assert isinstance(x, (Node, int, float)), f"The object {x} is not a Node, integer, or float"
    if isinstance(x, (int, float)):
        x = Node(x, )
    if (np.abs(np.cos(x.value)) < np.finfo(float).eps):
        raise ValueError("Invalid domain for Tan.")
    val = np.tan(x.value)
    deriv = (
        (x, 1/((np.cos(x.value))**2)),
    )
    return Node(val, list(deriv))

def exp(x):
    """
    Overloaded elementary exponential function

    Parameters
    =======
    x: must be Node, int, or float 

    Returns
    =======
    A new Node object with the derivative of the
    exponential computation done and the evaluation of the gradient

    EXAMPLES
    =======
    >>> x = Node(2)
    >>> f1 = rmo.exp(x)
    >>> f1.value
    7.38905609893065
    >>> f1.deriv
    [(Reverse-Mode AD: (f(x)=2, J=()), 7.38905609893065)]
    """
    assert isinstance(x, (Node, int, float)), f"The object {x} is not a Node, integer, or float"
    if isinstance(x, (int, float)):
        x = Node(x, )
    val = np.exp(x.value)
    deriv = (
        (x, np.exp(x.value)),
    )
    return Node(val, list(deriv))

def ln(x):
    """
    Overloaded elementary natural log function

    Parameters
    =======
    x: must be Node, int, or float 

    Returns
    =======
    A new Node object with the derivative of the
    natural log computation done and the evaluation of the gradient

    EXAMPLES
    =======
    >>> x = Node(2)
    >>> f1 = rmo.ln(x)
    >>> f1.value
    0.6931471805599453
    >>> f1.deriv
    [(Reverse-Mode AD: (f(x)=2, J=()), 0.5)]
    """
    assert isinstance(x, (Node, int, float)), f"The object {x} is not a Node, integer, or float"
    if isinstance(x, (int, float)):
        x = Node(x, )
    if x.value <= 0:
        raise ValueError("Cannot comput log of negative numbers or 0")
    val = np.log(x.value)
    deriv = (
        (x, 1/x.value),
    )
    return Node(val, list(deriv))

def log(x, base):
    """
    Overloaded elementary function log with a scalar base.

    Parameters
    =======
    x: must be Node, int, or float 

    Returns
    =======
    A new Node object with the derivative of the
    log base 'b' computation done and the evaluation of the gradient

    EXAMPLES
    =======
    >>> x = Node(2)
    >>> base = 5
    >>> f1 = rmo.log(x,base)
    >>> f1.value
    0.43067655807339306
    >>> f1.deriv
    [(Reverse-Mode AD: (f(x)=2, J=()), 0.31066746727980593)]
    """
    assert isinstance(x, (Node, int, float)), f"The object {x} is not a Node, integer, or float"
    if isinstance(x, (int, float)):
        x = Node(x, )
    if x.value <= 0:
        raise ValueError("Cannot comput log of negative numbers or 0")
    val = np.log(x.value)/np.log(base)
    deriv = (
        (x, 1/(x.value*np.log(base))),
    )
    return Node(val, list(deriv))

def arcsin(x):
    """
    Overloaded elementary trig function inverse sine

    Parameters
    =======
    x: must be Node, int, or float 

    Returns
    =======
    A new Node object with the derivative of the
    arcsin computation and the evaluation of the gradient

    EXAMPLES
    =======
    >>> x = Node(.05)
    >>> f1 = rmo.arcsin(x)
    >>> f1.value
    0.050020856805770016
    >>> f1.deriv
    [(Reverse-Mode AD: (f(x)=0.05, J=()), 1.0012523486435176)]
    """
    assert isinstance(x, (Node, int, float)), f"The object {x} is not a Node, integer, or float"
    if isinstance(x, (int, float)):
        x = Node(x, )
    if -1 > x.value or x.value > 1:
        raise ValueError("Invalid Domain, must be between -1 and 1")
    val = np.arcsin(x.value)
    deriv = (
        (x, 1/np.sqrt(1 - x.value**2)),
    )
    return Node(val, list(deriv))

def arccos(x):
    """
    Overloaded elementary trig function inverse cosine

    Parameters
    =======
    x: must be Node, int, or float 

    Returns
    =======
    A new Node object with the derivative of the
    arccos computation and the evaluation of the gradient

    EXAMPLES
    =======
    >>> x = Node(.05)
    >>> f1 = rmo.arccos(x)
    >>> f1.value
    1.5207754699891267
    >>> f1.deriv
    [(Reverse-Mode AD: (f(x)=0.05, J=()), -1.0012523486435176)]
    """
    assert isinstance(x, (Node, int, float)), f"The object {x} is not a Node, integer, or float"
    if isinstance(x, (int, float)):
        x = Node(x, )
    if -1 > x.value or x.value > 1:
        raise ValueError("Invalid Domain, must be between -1 and 1")
    val = np.arccos(x.value)
    deriv = (
        (x, -1/np.sqrt(1 - x.value**2)),
    )
    return Node(val, list(deriv))

def arctan(x):
    """
    Overloaded elementary trig function inverse tan

    Parameters
    =======
    x: must be Node, int, or float 

    Returns
    =======
    A new Node object with the derivative of the
    arctan computation and the evaluation of the gradient

    EXAMPLES
    =======
    >>> x = Node(.05)
    >>> f1 = rmo.arctan(x)
    >>> f1.value
    0.049958395721942765
    >>> f1.deriv
    [(Reverse-Mode AD: (f(x)=0.05, J=()), 0.9975062344139651)]
    """
    assert isinstance(x, (Node, int, float)), f"The object {x} is not a Node, integer, or float"
    if isinstance(x, (int, float)):
        x = Node(x, )
    val = np.arctan(x.value)
    deriv = (
        (x, 1/((x.value**2) + 1)),
    )
    return Node(val, list(deriv))

def sinh(x):
    """
    Overloaded elementary trig function sinh

    Parameters
    =======
    x: must be Node, int, or float 

    Returns
    =======
    A new Node object with the derivative of the
    sinh computation and the evaluation of the gradient

    EXAMPLES
    =======
    >>> x = Node(.05)
    >>> f1 = rmo.sinh(x)
    >>> f1.value
    0.050020835937655016
    >>> f1.deriv
    [(Reverse-Mode AD: (f(x)=0.05, J=()), 1.001250260438369)]
    """
    assert isinstance(x, (Node, int, float)), f"The object {x} is not a Node, integer, or float"
    if isinstance(x, (int, float)):
        x = Node(x, )
    val = np.sinh(x.value)
    deriv = (
        (x, np.cosh(x.value)),
    )
    return Node(val, list(deriv))

def cosh(x):
    """
    Overloaded elementary trig function cosh

    Parameters
    =======
    x: must be Node, int, or float 

    Returns
    =======
    A new Node object with the derivative of the
    cosh computation and the evaluation of the gradient

    EXAMPLES
    =======
    >>> x = Node(.05)
    >>> f1 = rmo.cosh(x)
    >>> f1.value
    1.001250260438369
    >>> f1.deriv
    [(Reverse - Mode AD: (f(x) = 0.05, J = ()), 0.050020835937655016)]
    """
    assert isinstance(x, (Node, int, float)), f"The object {x} is not a Node, integer, or float"
    if isinstance(x, (int, float)):
        x = Node(x, )
    val = np.cosh(x.value)
    deriv = (
        (x, np.sinh(x.value)),
    )
    return Node(val, list(deriv))

def tanh(x):
    """
    Overloaded elementary trig function tanh

    Parameters
    =======
    x: must be Node, int, or float 

    Returns
    =======
    A new Node object with the derivative of the
    tanh computation and the evaluation of the gradient

    EXAMPLES
    =======
    >>> x = Node(.05)
    >>> f1 = rmo.tanh(x)
    >>> f1.value
    0.04995837495787997
    >>> f1.deriv
    [(Reverse - Mode AD: (f(x) = 0.05, J = ()), 0.9975041607715679)]
    """
    assert isinstance(x, (Node, int, float)), f"The object {x} is not a Node, integer, or float"
    if isinstance(x, (int, float)):
        x = Node(x, )
    val = np.tanh(x.value)
    deriv = (
        (x, 1 - np.tanh(x.value)**2),
    )
    return Node(val, list(deriv))

def sigmoid(x):
    """
    Overloaded elementary trig function sigmoid

    Parameters
    =======
    x: must be Node, int, or float 

    Returns
    =======
    A new Node object with the derivative of the
    sigmoid computation and the evaluation of the gradient

    EXAMPLES
    =======
    >>> x = Node(.05)
    >>> f1 = rmo.sigmoid(x)
    >>> f1.value
    0.5124973964842103
    >>> f1.deriv
    [(Reverse-Mode AD: (f(x)=0.05, J=()), 0.24984381508111644)]
    """
    assert isinstance(x, (Node, int, float)), f"The object {x} is not a Node, integer, or float"
    if isinstance(x, (int, float)):
        x = Node(x, )
    val = 1/(1 + np.exp(-x.value))
    deriv = (
        (x, val*(1 - val)),
    )
    return Node(val, list(deriv))