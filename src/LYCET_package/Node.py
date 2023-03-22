#!/usr/bin/env python3
#File: Node.py
#Description: create Node class and overload operators to carry out reverse mode automatic differentiation

import numpy as np
from collections import defaultdict
    
class Node: 
    """
    A class to represent a Node object, or each variable in the computational graph.

    Attributes
    ----------
    value : int, float
        value of input x
    deriv : tuple
        child node and its partial derivative(s) (outer part of chain rule)

    Methods
    -------
    __eq__(other):
        equate Nodes
    __ne__(other):
        compare Nodes
    __lt__(other):
        compare Nodes
    __le__(other):
        compare Nodes
    __gt__(other):
        compare Nodes
    __ge__(other):
        compare Nodes
    __add__(other):
        add Nodes
    __sub__(other):
        subtract Nodes
    __neg__():
        negate Nodes
    __mul__(other):
        multiply Nodes
    __truediv__(other):
        divide Nodes 
    __pow__(other): 
        put Nodes to a power
    __radd__(other):
        reverse add Nodes
    __rsub__(other):
        reverse subtract Nodes
    __rmul__(other):
        reverse multiply Nodes
    __rtruediv__(other):
        reverse divide Nodes
    __repr__():
        string representation of Nodes
    Example
    -------
    >>> x = Node(4)
    >>> x.value
    4
    >>> x.deriv
    ()
    """

    def __init__(self, value, deriv=()):
        """
        Constructs all necessary attributes for the Node object.
        
        Parameters
        ----------
        value : int, float
            value of input x
        dual : tuple
            child node and its partial derivative(s) (outer part of chain rule)
        """
        assert isinstance(value, (int, float)), f"The value input {value} is not a integer, or float"
        self.value = value
        self.deriv = deriv

    def get_adjoints(self):
        """
        Recursively compute the adjoints.
        
        Parameters
        ----------
        none

        Returns
        -------
        adjoints: a dictionary

        Example
        -------
        >>> f = lambda x1, x2: rmo.ln(x1/x2)
        >>> x1 = Node(5)
        >>> x2 = Node(9)
        >>> x_nodes = [x1,x2]
        >>> f = f(*x_nodes)
        >>> Node.get_adjoints(f)
		{Reverse-Mode AD: (f(x)=0.5555555555555556, 
            J=[(Reverse-Mode AD: (f(x)=5, J=()), 0.1111111111111111), 
            (Reverse-Mode AD: (f(x)=9, J=()), -0.06172839506172839)]): 1.7999999999999998,
             Reverse-Mode AD: (f(x)=5, J=()): 0.19999999999999996,
             Reverse-Mode AD: (f(x)=9, J=()): -0.11111111111111109})
        """
        adjoints = defaultdict(int)

        def compute_adjoints(node, val):
            """
            Computes the adjoints.
            
            Parameters
            ----------
            node : parent node
            val : value of adjoint

            Returns
            -------
            adjoints : dictionary
                adjoints resulting from reverse pass 

            Example
            -------
            User does not directly call compute_adjoints.
            See get_adjoints for an example on how to retrieve the Jacobian.
            """
            for child, deriv in node.deriv:
                # calculate adjoint:
                vbar = val*deriv
                adjoints[child] += vbar
                # keep traversing through the graph
                compute_adjoints(child, vbar)

        compute_adjoints(self, 1) # end of forward pass
        return adjoints
         
    def __eq__(self, other):
        """
        Overload the equal operator to see if nodes are equal.
        
        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        bool: The return value. True for success, False otherwise.

        Example
        -------
        >>> x = Node(0)
        >>> y = 3
        >>> x == y
        False
        """
        if not isinstance(other, Node):
            other = Node(other, 0)
        if (self.deriv == () and other.deriv == ()):
            return (np.abs(self.value - other.value) < np.finfo(float).eps)
        else:
            return (np.abs(self.value - other.value) < np.finfo(float).eps) and (
                    np.abs(self.deriv - other.deriv) < np.finfo(float).eps)
    
    def __hash__(self):
        """
        Overload hash to be able to use Node type as key.
        
        Parameters
        -------
        none

        Returns
        -------
        Returns the hashed value if possible

        Example
        -------
	    >>> X = Node(2)
        >>> X.__hash__():
		2438534820432
	    >>> id(X)
 		2438534820432
        """
        return id(self)

    def __ne__(self, other):
        """
        Overload the not equal operator to see if nodes are not equal.
        
        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        bool: The return value. True for success, False otherwise.

        Example
        -------
        >>> x = Node(0)
        >>> y = Node(5)
        >>> x == y
        False
        """
        if not isinstance(other, Node):
            other = Node(other,0)
        if (self.deriv == () and other.deriv == ()) or (self.deriv == 0 and other.deriv == 0) :
            return not (np.abs(self.value - other.value) < np.finfo(float).eps)
        else:
            return not (np.abs(self.value-other.value)<np.finfo(float).eps) and (np.abs(self.deriv-other.deriv)<np.finfo(float).eps)

    def __lt__(self, other):
        """
        Overload the less than operator to compare two nodes

        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        bool: The return value. True for success, False otherwise.

        Example
        -------
        >>> x = Node(0)
        >>> y = 3
        >>> x < y
        True
        """
        if not isinstance(other, Node):
            other = Node(other,0)
        if (self.deriv == () and other.deriv == ()) or (self.deriv == 0 and other.deriv == 0):
            return np.less(self.value,other.value)
        else:
            return (np.less(self.value, other.value) and np.less(self.deriv, other.deriv))

    def __le__(self, other):
        """
        Overload the less than or equal operator to compare two nodes
        
        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        bool: The return value. True for success, False otherwise.

        Example
        -------
        >>> x = Node(3)
        >>> y = 3
        >>> x <= y
        True
        """
        if not isinstance(other, Node):
            other = Node(other, 0)
        if self.deriv == () and other.deriv == ():
            return np.less_equal(self.value, other.value)
        else:
            return (np.less_equal(self.value, other.value) and np.less_equal(self.deriv, other.deriv))

    def __gt__(self, other):
        """
        Overload the greater operator to compare two nodes
        
        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        bool: The return value. True for success, False otherwise.

        Example
        -------
        >>> x = Node(3)
        >>> y = Node(7)
        >>> x > y
        False
        """
        if not isinstance(other, Node):
            other = Node(other, 0)
        if (self.deriv == () and other.deriv == ()) or (self.deriv == 0 and other.deriv == 0):
            return np.greater(self.value, other.value)
        else:
            return (np.greater(self.value, other.value) and np.greater(self.deriv, other.deriv))

    def __ge__(self, other):
        """
        Overload the greater than or equal operator to compare two nodes
        
        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        bool: The return value. True for success, False otherwise.

        Example
        -------
        >>> x = Node(3)
        >>> y = Node(7)
        >>> x >= y
        False
        """
    
        if not isinstance(other, Node):
            other = Node(other, 0)
        if (self.deriv == () and other.deriv == ()):
            return np.greater_equal(self.value, other.value)
        else:
            return (np.greater_equal(self.value, other.value) and np.greater_equal(self.deriv, other.deriv))
            
    def __add__(self, other):
        """
        Create a new node that is the sum of the two previous nodes.
        
        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        Node

        Example
        -------
	    >>> X1 = Node(5)
	    >>> X2 = Node(6)
	    >>> x3 = X2+X1
	    >>> x3.value
   		11
	    >>> x3.deriv
    	((Reverse-Mode AD: (f(x)=6, J=()), 1), (Reverse-Mode AD: (f(x)=5, J=()), 1))
        """
        assert isinstance(other, (Node,int,float)), f'input {other} is not a Node, int, or float'
        if isinstance(other, (int, float)): # other is a constant
            other = Node(other, 0) 
        value = self.value + other.value
        deriv = ((self, 1), (other, 1))# the partial derivative with respect to x1 is 1, the partial derivative with respect to x2 is 1
        
        return Node(value, deriv)

    def __mul__(self, other):
        """
        Create a new node that is the product of the two previous nodes.
        
        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        Node

        Example
        -------
	    >>> X1 = Node(5)
	    >>> X2 = Node(6)
	    >>> x3 = X2*X1
	    >>> x3.value
    	30
	    >>> x3.deriv
   		((Reverse-Mode AD: (f(x)=6, J=()), 5), (Reverse-Mode AD: (f(x)=5, J=()), 6))
        """
        assert isinstance(other, (Node,int,float)), f'input {other} is not a Node'
        if isinstance(other, (int, float)):
           other = Node(other, )
        value = self.value * other.value
        deriv = ((self, other.value), (other, self.value))

        return Node(value, deriv)

    def __sub__(self, other): 
        """
        Create a new node that is the difference of the two previous nodes.
        
        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        Node

        Example
        -------
	    >>> X1 = Node(5)
	    >>> X2 = Node(6)
	    >>> x3 = X2-X1
	    >>> x3.value
   		1
	    >>> x3.deriv
		((Reverse-Mode AD: (f(x)=6, J=()), 1), (Reverse-Mode AD: (f(x)=5, J=()), -1))   
        """
        assert isinstance(other, (Node,int,float)), f'input {other} is not a Node'
        if isinstance(other, (int, float)):
            other = Node(other, 0)
        value = self.value - other.value
        deriv = ((self, 1), (other, -1))
        return Node(value, deriv)

    def __truediv__(self, other): 
        """
        Create a new node that is the quotient of the two previous nodes.
        
        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        Node

        Example
        -------
   	    >>> X1 = Node(5)
	    >>> X2 = Node(6)
	    >>> x3 = X2/X1
	    >>> x3.value
    	1.2
	    >>> x3.deriv
    	[(Reverse - Mode AD: (f(x) = 6, J = ()), 0.2), (Reverse - Mode AD: (f(x) = 5, J = ()), -0.24)]
        """
        assert isinstance(other, (Node, int, float)), f"The object {other} is not a Node, integer, or float"
        if isinstance(other, (int, float)):
            other = Node(other, 0)
        value = self.value / other.value
        deriv = ((self, 1/other.value), (other, -1*self.value/(other.value**2)))
        return Node(value, list(deriv))

    def __pow__(self, other): 
        """
        Take the power of x raised to Node other.
        
        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        Node

        Example
        -------
	    >>> X1 = Node(5)
	    >>> X2 = 3
	    >>> x3 = X1**X2
	    >>> x3.value
    	125
	    >>> x3.deriv
    	((Reverse-Mode AD: (f(x)=5, J=()), 75),)
        """
        assert isinstance(other, (Node, int, float)), f"The object {other} is not a Node, integer, or float"
        if isinstance(other, (int, float)):
            other = Node(other, 0)
        value = self.value ** other.value
        deriv = (
            (self, other.value*(self.value**(other.value-1))), 
        )
        return Node(value, deriv)

    def __radd__(self, other):
        """
        Overload the reverse addition operator to find the sum of two nodes

        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        Node

        Example
        -------
        >>> X1 = Node(5)
	    >>> X2 = Node(6)
	    >>> x3 = X1+X2
	    >>> x3.value
   		11
	    >>> x3.deriv
    	((Reverse-Mode AD: (f(x)=6, J=()), 1), (Reverse-Mode AD: (f(x)=5, J=()), 1))
        """
        return self.__add__(other)

    def __rsub__(self, other):
        """
        Overload the reverse subtraction operator to find the difference of two nodes

        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        Node

        Example
        -------
        >>> X1 = Node(5)
	    >>> X2 = Node(6)
	    >>> x3 = X1-X2
	    >>> x3.value
   		-1
	    >>> x3.deriv
        ((Reverse-Mode AD:((f(x)=5, J=()), 1), Reverse-Mode AD: ((f(x)=6, J=()), -1)))
        """
        return self.__sub__(other)

    def __rmul__(self, other):
        """
        Overload the reverse multiplication operator to find the product of two nodes

        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        Node

        Example
        -------
	    >>> X1 = Node(5)
	    >>> X2 = Node(6)
	    >>> x3 = X1*X2
	    >>> x3.value
    	30
	    >>> x3.deriv
   		((Reverse-Mode AD: (f(x)=5, J=()), 6), (Reverse-Mode AD: (f(x)=6, J=()), 5))
        """
        return self.__mul__(other)

    def __rtruediv__(self, other):
        """
        Overload the reverse division operator to divide two node types.

        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        Node

        Example
        -------

        >>> x1 = Node(9)
        >>> x2 = Node(3)
        >>> f = x1/x2
        >>> f.value 
       	3
        >>> f.deriv 
        [(Reverse-Mode AD: (f(x)=9, J=()), 0.3333333333333333), (Reverse-Mode AD: (f(x)=3, J=()), -1.0)]
        """
        assert np.issubdtype(type(other), np.integer) or isinstance(other, (np.floating, float)), f"The object {other} is not an integer or float" # check if number is a node, int or float
        
        if (np.abs(self.value) < np.finfo(float).eps):
            raise ZeroDivisionError('Cannot divide by zero. Node divisor has a real part of zero')

        if isinstance(other, (int, float)):
            other = Node(other, )
        value = other.value / self.value
        deriv = ((other, 1/self.value), (self, -1*other.value/(self.value**2)))
        return Node(value, list(deriv))


    def __rpow__(self, other):
        """
        Take the reverse power of node type.
        
        Parameters
        ----------
        other: Node, int, float
            
        Returns
        -------
        Node

        Example
        -------
	    >>> X1 = Node(5)
	    >>> X2 = 3
	    >>> x3 = X2**X1
	    >>> x3.value
    	243
	    >>> x3.deriv
    	((Reverse-Mode AD: (f(x)=3, J=()), 405),)
        """
        assert np.issubdtype(type(other), np.integer) or isinstance(other, (np.floating, float)), f"The object {other} is not an integer or float"
        if (np.abs(other) < np.finfo(float).eps):
            raise ValueError('Cannot divide by zero or compute logarithm of zero or both')
        if isinstance(other, (int, float)):
            other = Node(other, )
        value = other.value ** self.value
        deriv = (
            (other, self.value*(other.value**(self.value-1))), 
        )
        return Node(value, deriv)

    def __repr__(self):
        """
        Represents the class's objects as strings.

        Parameters
        ----------
        None
            
        Returns
        -------
        Representation of Node as a string

        Example
        -------
        >>> x1 = Node(9)
        >>> x2 = Node(3)
        >>> f = x1/x2
        >>> f.value 
        3
        >>> f.deriv 
        1/3, -1
        >>> print(repr(x))
        Reverse-Mode AD: (f(x)=3, J=[1/3, -1])
        """
        return f"Reverse-Mode AD: (f(x)={self.value}, J={self.deriv})"
    