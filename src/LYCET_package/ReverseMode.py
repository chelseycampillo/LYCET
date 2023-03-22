#!/usr/bin/env python3
# File: ReverseMode.py
# Description: function that user interfaces with to carry out reverse mode automatic differentiation

from .Node import Node

def ReverseMode(f, x):
    """
    Function that user interfaces with to compute the Jacobian of their complex function.

    Parameters
    ----------
    f : user defined function with reverse LYCET operations
    x : input variable(s)

    Output
    ------
    f.value : f evaluated at x
    J : Jacobian evaluated at x

    EXAMPLE
    -------
    >>> import LYCET_package.ReverseMode as rm
    >>> from LYCET_package.Node import Node
    >>> import LYCET_package.LYCET_Operations_Reverse as rmo
    >>> f = lambda x1, x2, x3: rmo.cos(x1 + x2) + (x3 * x2 ** 3)
    >>> x = [1, 2, 3]
    >>> ad_func = rm.ReverseMode(f, x)
    >>> ad_func[0]
    23.010007503399553 --> function evaluated at x1,x2,x3
    >>> ad_funct[1]
    [-0.1411200080598672, 35.858879991940135, 8] --> gradient of f 
    """
    if isinstance(x, (int, float)):
        x = [x]
    nodes = []
    for i in range(len(x)):
        node = Node(x[i])
        nodes.append(node)
    f = f(*nodes) # unpack list
    df = Node.get_adjoints(f)
    J = []
    for i in range(len(x)):
        ji = df[nodes[i]]
        J.append(ji)
    return f.value, J