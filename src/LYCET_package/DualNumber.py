#!/usr/bin/env python3
# File: DualNumber.py
# Description: Create dual number for forward mode of AD 

import numpy as np

class DualNumber:

    """
    A class to represent a DualNumber.

    Attributes
    ----------
    real : int, float
        the value at which to evaluate f(x) at
    dual : int or float
        the derivative of f(x)

    Methods
    -------
    __eq__(num):
        equate dual numbers
    __ne__(num):
        compare dual numbers
    __lt__(num):
        compare dual numbers
    __le__(num):
        compare dual numbers
    __gt__(num):
        compare dual numbers
    __ge__(num):
        compare dual numbers
    __add__(num):
        add dual numbers
    __sub__(num):
        subtract dual numbers
    __neg__():
        negate dual numbers
    __mul__(num):
        multiply dual numbers
    __truediv__(num):
        divide dual numbers 
    __floordiv__(num):
        divide dual numbers
    __pow__(num): 
        put dual numbers to a power
    __radd__(num):
        reverse add dual numbers
    __rsub__(num):
        reverse subtract dual numbers
    __rmul__(num):
        reverse multiply dual numbers
    __rtruediv__(num):
        reverse divide dual numbers
    __rfloordiv__(num):
        reverse divide dual numbers
    __repr__():
        string representation of dual numbers

    Example
    -------
    >>> x = DualNumber(4)
    >>> x.real
    4
    >>> x.dual
    1
    """

    def __init__(self, real, dual=1.0):
        """
        Constructs all the necessary attributes for the DualNumber object.

        Parameters
        ----------
        real : int, float
            the value at which to evaluate f(x) at
        dual : int or float
            the derivative of f(x)
        """
        self.real = real 
        self.dual = dual 

    def __eq__(self, num):
        """
        Overload the equal operator to equate two dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        bool: The return value. True for success, False otherwise.

        Example
        -------
        >>> x = 3 + DualNumber(0,0)
        >>> y = 3
        >>> x == y
        True
        """
        if not isinstance(num, DualNumber):
            num = DualNumber(num,0)

        return (np.abs(self.real-num.real)<np.finfo(float).eps) and (np.abs(self.dual-num.dual)<np.finfo(float).eps)

    def __ne__(self, num):
        """
        Overload the not equal operator to see if dual numbers are not equal

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        bool: The return value. True for success, False otherwise.

        Example
        -------
        >>> x = 3 + DualNumber(0,0)
        >>> y = DualNumber(5,4)
        >>> x == y
        False
        """
        return not self.__eq__(num)

    def __lt__(self, num):
        """
        Overload the less than operator to compare two dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        bool: The return value. True for success, False otherwise.

        Example
        -------
        >>> x = DualNumber(0,0)
        >>> y = 3
        >>> x < y
        True
        """
        if not isinstance(num, DualNumber):
            num = DualNumber(num,0)

        return (np.less(self.real, num.real) and np.less(self.dual, num.dual)) 

    def __le__(self, num):
        """
        Overload the less than or equal to operator to compare two dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        bool: The return value. True for success, False otherwise.

        Example
        -------
        >>> x = DualNumber(3,0)
        >>> y = 3
        >>> x <= y
        True
        """
        if not isinstance(num, DualNumber):
            num = DualNumber(num,0)

        return (np.less_equal(self.real, num.real) and np.less_equal(self.dual, num.dual)) 

    def __gt__(self, num):
        """
        Overload the greater than operator to compare two dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        bool: The return value. True for success, False otherwise.

        Example
        -------
        >>> x = DualNumber(0,0)
        >>> y = 3
        >>> x > y
        False
        """
        if not isinstance(num, DualNumber):
            num = DualNumber(num,0)

        return (np.greater(self.real, num.real) and np.greater(self.dual, num.dual)) 

    def __ge__(self, num):
        """
        Overload the greater than or equal to operator to compare two dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        bool: The return value. True for success, False otherwise.

        Example
        -------
        >>> x = DualNumber(0,0)
        >>> y = 3
        >>> x >= y
        False
        """
        if not isinstance(num, DualNumber):
            num = DualNumber(num,0)

        return (np.greater_equal(self.real, num.real) and np.greater_equal(self.dual, num.dual)) 

    def __add__(self, num):
        """
        Overload the addition operator to find the sum of dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        DualNumber

        Example
        -------
        >>> x = DualNumber(1,2)
        >>> x + 3
        DualNumber(4,2)
        """
        # check if number is a dual number, int or float
        assert isinstance(num, (DualNumber)) or np.issubdtype(type(num), np.integer) or isinstance(num, (np.floating, float)), f"The object {num} is not a Dual Number, integer or float"
        
        if isinstance(num, DualNumber):
            return DualNumber(self.real + num.real, self.dual + num.dual)
        
        return DualNumber(self.real + num, self.dual)

    def __sub__(self, num):
        """
        Overload the subtraction operator to find the difference of dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        DualNumber

        Example
        -------
        >>> x = DualNumber(1,2)
        >>> x - 3
        DualNumber(-2,2)
        """
        # check if number is a dual number, int or float
        assert isinstance(num, (DualNumber)) or np.issubdtype(type(num), np.integer) or isinstance(num, (np.floating, float)), f"The object {num} is not a Dual Number, integer or float"
        
        if isinstance(num, DualNumber):
            return DualNumber(self.real - num.real, self.dual - num.dual)
        
        return DualNumber(self.real - num, self.dual)
        
    def __neg__(self):
        """
        Overload the negation operator to negate of dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        DualNumber

        Example
        -------
        >>> x = DualNumber(1,2)
        >>> -x
        DualNumber(-1,-2)
        """
        return DualNumber(-self.real, -self.dual)

    def __mul__(self, num):
        """
        Overload the multiplication operator to multiply dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        DualNumber

        Example
        -------
        >>> x = DualNumber(1,2)
        >>> y = DualNumber(1,2)
        >>> x*y
        DualNumber(1,4)
        """
        # check if number is a dual number, int or float
        assert isinstance(num, (DualNumber)) or np.issubdtype(type(num), np.integer) or isinstance(num, (np.floating, float)), f"The object {num} is not a Dual Number, integer or float"
        
        if isinstance(num, DualNumber):
            return DualNumber(self.real * num.real, self.real*num.dual + self.dual*num.real)
        
        return DualNumber(self.real*num, self.dual*num)


    def __truediv__(self, num):
        """
        Overload the division operator to divide dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        DualNumber

        Example
        -------
        >>> x = DualNumber(6,9)
        >>> y = 3
        >>> x/y
        DualNumber(2,3)
        """
        # check if number is a dual number, int or float
        assert isinstance(num, (DualNumber)) or np.issubdtype(type(num), np.integer) or isinstance(num, (np.floating, float)), f"The object {num} is not a Dual Number, integer or float"
        
        if isinstance(num, DualNumber):
            if (num.real == 0):
                raise ZeroDivisionError('Cannot divide by zero. Dual number divisor has a real part of zero')
            return DualNumber(self.real/num.real, (self.dual * num.real - self.real * num.dual)/(num.real**2))
        
        else:
            if (np.abs(num) < np.finfo(float).eps):
                raise ZeroDivisionError('Cannot divide by zero. Scalar divisor is zero')
            return DualNumber(self.real/num, self.dual/num)

            
    def __floordiv__(self, num):
        """
        Overload the floor division operator to divide dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        DualNumber

        Example
        -------
        >>> x = DualNumber(6.1,9.1)
        >>> y = 3
        >>> x/y
        DualNumber(2,3)
        """
        # check if number is a dual number, int or float
        assert isinstance(num, (DualNumber)) or np.issubdtype(type(num), np.integer) or isinstance(num, (np.floating, float)), f"The object {num} is not a Dual Number, integer or float"
        
        if isinstance(num, DualNumber):
            if (num.real == 0):
                raise ZeroDivisionError('Cannot divide by zero. Dual number divisor has a real part of zero')
            return DualNumber(self.real//num.real, (self.dual * num.real - self.real * num.dual)//(num.real**2))
        
        else:
            if (np.abs(num) < np.finfo(float).eps):
                raise ZeroDivisionError('Cannot divide by zero. Scalar divisor is zero')
            return DualNumber(self.real//num, self.dual//num)


    def __pow__(self, num): 
        """
        Overload the power dunder method for dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        DualNumber

        Example
        -------
        >>> x = DualNumber(1,2)
        >>> x**3
        DualNumber(1,6)
        """
      
        assert isinstance(num, (DualNumber)) or np.issubdtype(type(num), np.integer) or isinstance(num, (np.floating, float)), f"The object {num} is not a Dual Number, integer or float"
        if isinstance(num, DualNumber):
            if (np.abs(self.real) < np.finfo(float).eps):
                if ((self.dual == 0) and (num.real > 0)) or ((self.dual != 0) and (num.real > 1)):
                    return DualNumber(0, 0)
                raise ValueError('Cannot divide by zero or compute logarithm of zero or both')
            return DualNumber(self.real**num.real, (self.real**(num.real-1))*(self.real*num.dual*np.log(self.real) + num.real*self.dual))
        
        else:
            if (np.abs(self.real) < np.finfo(float).eps) and (num < 1):
                raise ZeroDivisionError('Cannot divide by zero. Base dual number has a real part of zero and Exponent scalar is lower than 1: real part or dual part or both have a division by zero')
            return DualNumber(self.real**num, num*self.dual*(self.real**(num-1)))

    def __radd__(self, num):
        """
        Overload the reverse addition operator to find the sum of dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        DualNumber

        Example
        -------
        >>> x = DualNumber(1,2)
        >>> 3 + x
        DualNumber(4,2)
        """
        return self.__add__(num)

    def __rsub__(self, num):
        """
        Overload the reverse subtraction operator to find the difference of dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        DualNumber

        Example
        -------
        >>> x = DualNumber(1,2)
        >>> 3 - x
        DualNumber(-2,2)
        """
        return - self.__sub__(num)

    def __rmul__(self, num):
        """
        Overload the reverse multiplication operator to multiply dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        DualNumber

        Example
        -------
        >>> x = DualNumber(1,2)
        >>> y = DualNumber(1,2)
        >>> y*x
        DualNumber(1,4)
        """
        return self.__mul__(num)
    
    def __rtruediv__(self, num):
        """
        Overload the reverse division operator to divide dual numbers.

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        DualNumber

        Example
        -------
        >>> x = DualNumber(6,9)
        >>> y = 3
        >>> 3/y
        DualNumber(0.5,0.33)
        """
        # check if number is a dual number, int or float
        assert np.issubdtype(type(num), np.integer) or isinstance(num, (np.floating, float)), f"The object {num} is not an integer or float"
        
        if (np.abs(self.real) < np.finfo(float).eps):
            raise ZeroDivisionError('Cannot divide by zero. Dual number divisor has a real part of zero')
        return DualNumber(num/self.real, (-num*self.dual)/(self.real**2))
    
    def __rfloordiv__(self, num):
        """
        Overload the reverse floor division operator to divide dual numbers.

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        DualNumber

        Example
        -------
        >>> x = DualNumber(6.1,9.1)
        >>> y = 3
        >>> y/x
        DualNumber(0,0)
        """
        # check if number is a dual number, int or float
        assert np.issubdtype(type(num), np.integer) or isinstance(num, (np.floating, float)), f"The object {num} is not an integer or float"
        
        if (np.abs(self.real) < np.finfo(float).eps):
            raise ZeroDivisionError('Cannot divide by zero. Dual number divisor has a real part of zero')
        return DualNumber(num//self.real, (-num*self.dual)//(self.real**2))

    def __rpow__(self, num):
        """
        Overload the power dunder method for dual numbers

        Parameters
        ----------
        num : DualNumber, int, float
            
        Returns
        -------
        DualNumber

        Example
        -------
        >>> x = DualNumber(1,2)
        >>> 3**x
        DualNumber(3, 6*np.log(3))
        """
        assert np.issubdtype(type(num), np.integer) or isinstance(num, (np.floating, float)), f"The object {num} is not an integer or float"
        if (np.abs(num) < np.finfo(float).eps):
            if (self.real > 0):
                return DualNumber(0, 0)
            raise ValueError('Cannot divide by zero or compute logarithm of zero or both')
        return DualNumber(num**self.real, self.dual*np.log(num)*(num**self.real))

    def __repr__(self):
        """
        Represents the class's objects as strings.

        Parameters
        ----------
        None
            
        Returns
        -------
        Representation of DualNumber as a string

        Example
        -------
        >>> x = DualNumber(4,3)
        >>> print(repr(x))
        Dual Number (real=4, dual=3)
        """
        return f"Dual Number (real={self.real}, dual={self.dual})"

