#!/usr/bin/env python3
# File: ForwardMode.py
# Description: Arbitrary function to wrap all forward mode operations

import numpy as np
from .DualNumber import DualNumber


def ForwardMode(f, x, p=None, gradient=False, jacobian=False):
    """
    Function that user interfaces with to compute scalar/vector functions with scalar/vector inputs of their complex function.

    Parameters
    ----------
    f : user defined function with forward LYCET operations
    x : input variable(s)
    p : optional 
        seed vector
    gradient : optional
    jacobian : optional

    Output
    ------
    if p == None:
        f  : f evaluated at the specified input
        df : df evaluated at the specified input
    if p != None: 
        f : f evaluated at the specified input
        Df: directional directional based on p
    if gradient == True:
        return only gradient
    if jacobian == True:
        return only Jacobian 

    EXAMPLE
    -------
    def user_func(x):
        return lop.exp(x) + lop.sin(lop.exp(x))

    >>> x = 4
    >>> f, df = lop.ForwardMode(user_func, x)
    >>> print(f, df)
    (53.66938209690045, 34.360705101546074)
    """
    if not (gradient or jacobian): 
        if p is None: # Unidimentional
            assert isinstance(x, (DualNumber)) or np.issubdtype(type(x), np.integer) or isinstance(x, (np.floating, float)) or ((type(x) in [tuple, list, np.ndarray]) and (len(x)==1)), f"{x}, {type(x)} has to be a DualNumber, an integer, a float or an array-like of length one in the unidimentional case"
            if ((type(x) in [tuple, list, np.ndarray]) and (len(x)==1)):
                x = x[0]
                x = DualNumber(x)
                return f([x]).real, f([x]).dual
            x = DualNumber(x)
            if np.issubdtype(type(f(x)), np.integer) or isinstance(f(x), (np.floating, float)):
                #when the multidimensional case goes here, 
                #it means that we try to take the deirvative of f w.r.t a variable that does not appear in the expression of f
                #e.g. deivative of f(x1, x2, x3) = x1*cos(x2) w.r.t x3
                return f(x), 0
            return f(x).real, f(x).dual 

        else: # Multidimentional input
                assert np.issubdtype(type(f(x)), np.integer) or isinstance(f(x), (np.floating, float)) or (type(f(x)) in [list, tuple, np.ndarray]), f"output {f(x)} has to be an integer, float, list, tuple or np.ndarray"
                if (type(f(x)) in [list, tuple, np.ndarray]):
                    assert np.all([np.issubdtype(type(k), np.integer) or isinstance(k, (np.floating, float)) for k in f(x)]), f"{f(x)} has to contain only floats or integers"
                #assert (type(p) in [list, tuple, np.ndarray]) and (len(x)==len(p)) , f"{p} has to be a list, tuple or np.ndarray of same length as {x}"
                assert (type(p) in [list, tuple, np.ndarray]), f"{p} has to be a list, tuple or np.ndarray"
                assert np.all([np.issubdtype(type(k), np.integer) or isinstance(k, (np.floating, float)) for k in p]), f"{p} has to contain only floats or integers"
                if (len(p) == 1):
                    return ForwardMode(f, x)
                else:
                    assert (len(p) == len(x)), f"{p} must have same length as {x}"
                    assert (type(x) in [list, tuple, np.ndarray]), f"{x} has to be a list, tuple or np.ndarray in the multidimensional case"
                    assert np.all([np.issubdtype(type(k), np.integer) or isinstance(k, (np.floating, float)) for k in x]), f"{x} has to contain only floats or integers"

                    x = np.array(x)
                    p = np.array(p)

                    if np.issubdtype(type(f(x)), np.integer) or isinstance(f(x), (np.floating, float)): # function at values in R

                        # get the indexes of non zero values in seed vector p
                        # the output will be the directional derivative of f, which is equal to the scalar product between gradient and p
                        # in fact, the result is a linear combination of product between partial derivatives and non zero values in seed vector p
                        non_zero_indexes = [(i,v) for i,v in enumerate(p) if v!=0]
                        partial_derivatives = []
                        for j,v in non_zero_indexes:
                            new_f = lambda y : f([y if (i==j) else val for i,val in enumerate(x)])
                            deriv = ForwardMode(new_f, x[j])[1] # dual part of output DualNumber
                            partial_derivatives.append(v*deriv)
                        return f(x), sum(partial_derivatives)  
                    else: # function at values in R^m
                        # use ForwardMode on coordinate functions to get the Jacobian
                        if np.issubdtype(type(x), np.integer) or isinstance(x, (np.floating, float)):
                            nb_var = 1
                        else:
                            nb_var = len(x)
                        nb_func = len(f(x))
                        jacobian = np.zeros((nb_func, nb_var))
                        for j in range(nb_var): #goes through variables
                            for i in range(nb_func): #goes through coordinates functions
                                coord_func = lambda y : f(y)[i]
                                new_p = [1 if (k==j) else 0 for k in range(nb_var)]
                                deriv = ForwardMode(coord_func, x, p=new_p)[1]
                                jacobian[i,j] = deriv
                        p = p.reshape(-1,1)
                        res = jacobian @ p
                        return np.array(f(x)), res.reshape(1,-1)[0]

    else: #return either the gradient or the jacobian
        
        assert np.issubdtype(type(x), np.integer) or isinstance(x, (np.floating, float)) or (type(x) in [list, tuple, np.ndarray]), f"input {x} has to be an integer, float, list, tuple or np.ndarray"
        if (type(x) in [list, tuple, np.ndarray]):
            assert np.all([np.issubdtype(type(k), np.integer) or isinstance(k, (np.floating, float)) for k in x]), f"{x} has to contain only floats or integers"
        assert np.issubdtype(type(f(x)), np.integer) or isinstance(f(x), (np.floating, float)) or (type(f(x)) in [list, tuple, np.ndarray]), f"output {f(x)} has to be an integer, float, list, tuple or np.ndarray"
        if (type(f(x)) in [list, tuple, np.ndarray]):
            assert np.all([np.issubdtype(type(k), np.integer) or isinstance(k, (np.floating, float)) for k in f(x)]), f"{f(x)} has to contain only floats or integers"
            
        if gradient: #even if jacobian is also True, in this case jacobian = gradient so we only return the gradient
            if not (np.issubdtype(type(f(x)), np.integer) or isinstance(f(x), (np.floating, float))):
                raise ValueError(f"Cannot compute gradient if function is not at values in a unidimentional space. {f(x)} of type {type(x)} should be an integer or a float")
            if np.issubdtype(type(x), np.integer) or isinstance(x, (np.floating, float)) or ((type(x) in [list, tuple, np.ndarray]) and (len(x)==1)): # function R -> R
                if ((type(x) in [list, tuple, np.ndarray]) and (len(x)==1)):
                    x = x[0]
                    x = DualNumber(x)
                    return f([x]).dual
                x = DualNumber(x)
                return f(x).dual
            else: # function R^n -> R
                assert np.all([np.issubdtype(type(k), np.integer) or isinstance(k, (np.floating, float)) for k in x]), f"{x} has to contain only floats or integers"
                nb_var = len(x)
                list_p = []
                for i in range(nb_var):
                    list_p.append([1 if (k==i) else 0 for k in range(nb_var)])
                grad = []
                for new_p in list_p:
                    deriv = ForwardMode(f, x, p=new_p)[1]
                    grad.append(deriv)
                return np.array(grad)
            
        if jacobian:
            if not (np.issubdtype(type(f(x)), np.integer) or isinstance(f(x), (np.floating, float))): # at values in R^m
                if np.issubdtype(type(x), np.integer) or isinstance(x, (np.floating, float)):
                    nb_var = 1
                else:
                    nb_var = len(x)
                nb_func = len(f(x)) 
                jacobian = np.zeros((nb_func, nb_var))
                for j in range(nb_var): #goes through variables
                    for i in range(nb_func): #goes through coordinates functions
                        coord_func = lambda y : f(y)[i]
                        new_p = [1 if (k==j) else 0 for k in range(nb_var)]
                        deriv = ForwardMode(coord_func, x, p=new_p)[1]
                        jacobian[i,j] = deriv
                return jacobian
            else:
                return ForwardMode(f, x, gradient=True)