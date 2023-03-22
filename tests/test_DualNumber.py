import pytest
import math

from LYCET_package.DualNumber import DualNumber

class TestDualNumber:

    def test_init(self):
        # Construction
        x = DualNumber(2,3)
        y = DualNumber(2,3)

        assert y == x

    def test_add(self):
        # add (dual, non-dual)
        x = DualNumber(1,2)
        assert x + 3 == DualNumber(1+3,2)

    def test_add_dual(self):
        # add (dual, dual)
        x = DualNumber(1,2)
        assert x + DualNumber(3,5) == DualNumber(1+3,2+5)

    def test_radd(self):
        # add (non-dual, dual)
        x = DualNumber(1,2)
        assert 3 + x == DualNumber(1+3,2)

    def test_sub(self):
        # subtract (dual, non-dual)
        x = DualNumber(1,2)
        assert x - 3 == DualNumber(1-3,2)

    def test_sub_dual(self):
        # subtract (dual, dual)
        x = DualNumber(1,2)
        assert x - DualNumber(3,5) == DualNumber(1-3,2-5)

    def test_rsub(self):
        # subtract (non-dual, dual)
        x = DualNumber(1,2)
        assert 3 - x == DualNumber(3-1,-2)

    def test_mul(self):
        # multiply (dual, non-dual)
        x = DualNumber(1, 2)
        assert x * 3 == DualNumber(1*3,2*3)

    def test_mul_dual(self):
        # multiply (dual, dual)
        x = DualNumber(1, 2)
        assert x * DualNumber(3,5) == DualNumber(1*3,1*5+2*3)

    def test_rmul(self):
        # multiply (non-dual, dual)
        x = DualNumber(1, 2)
        assert 3 * x == DualNumber(1*3,2*3)

    def test_truediv(self):
        # divide (dual, non-dual)
        x = DualNumber(1, 2)
        assert x / 3 == DualNumber(1/3, 2/3)

    def test_truediv_zero(self):
        # divide (dual, non-dual == 0)
        x = DualNumber(1, 2)
        with pytest.raises(ZeroDivisionError):
            x / 0

    def test_truediv_dual(self):
        # divide (dual, dual)
        x = DualNumber(1, 2)
        assert x / DualNumber(3,5) == DualNumber(1/3, (2*3 - 1*5)/(3*3) )

    def test_truediv_zero_dual(self):
        # divide (dual, dual with real part 0)
        x = DualNumber(1, 2)
        with pytest.raises(ZeroDivisionError):
            x / DualNumber(0,3)

    def test_rtruediv(self):
        # divide (non-dual, dual)
        x = DualNumber(1, 2)
        assert 3 / x == DualNumber(3/1, (-3*2)/(1*1))

    def test_rtruediv_zero_dual(self):
        # divide (non-dual, dual with real part 0)
        with pytest.raises(ZeroDivisionError):
            3 / DualNumber(0,2)

    def test_pow(self):
        # power (dual, non-dual)
        x = (DualNumber(4,3)) #
        assert x**2 ==  DualNumber(4**2,3*2*4) #f(a+b*eps) = f(a)+b*f'(a)*eps

    def test_pow_zero(self):
        # power (dual with real part 0, non-dual)
        x = (DualNumber(0,3)) #
        assert x**2 ==  DualNumber(0**2,3*2*0) #f(a+b*eps) = f(a)+b*f'(a)*eps
        with pytest.raises(ZeroDivisionError):
            x**0.5

    def test_pow_dual(self):
        # power (dual, dual)
        #
        # (a + b e) ^ (c + d e) = a^c * (1 + (d * ln(a) + b * c / a) e)
        a = 2
        b = 3
        c = 5
        d = 7
        assert DualNumber(a, b)**DualNumber(c, d) == DualNumber(a**c, a**c * (d * math.log(a) + b * c / a))

    def test_pow_zero_dual(self):
        # power (dual with real part 0, dual)
        #
        # (a + b e) ^ (c + d e) = a^c * (1 + (d * ln(a) + b * c / a) e)
        # = a^c + (a^c * d * ln(a) + a^(c - 1) * b * c) e
        # This is always undefined if a = 0 and c <= 0. For a = 0 and c > 0, there's a few possibilities:
        # * b = 0 and d = 0: simply the real number case, so the result should be zero for c > 0.
        # * b = 0 and d != 0: dual part is a^c * d * ln(a), which is zero for c > 0
        # * b != 0 and d = 0: dual part is a^(c - 1) * b * c, which is zero for c > 1
        # * b != 0 and d != 0: dual part is the sum of the above, so zero for c > 1
        assert DualNumber(0, 0)**DualNumber(0.5, 0) == DualNumber(0, 0)
        assert DualNumber(0, 0)**DualNumber(0.5, 3) == DualNumber(0, 0)
        assert DualNumber(0, 3)**DualNumber(1.5, 0) == DualNumber(0, 0)
        assert DualNumber(0, 3)**DualNumber(1.5, 5) == DualNumber(0, 0)
        with pytest.raises(ValueError):
            DualNumber(0, 0)**DualNumber(-0.5, 0)
        with pytest.raises(ValueError):
            DualNumber(0, 0)**DualNumber(-0.5, 3)
        with pytest.raises(ValueError):
            DualNumber(0, 3)**DualNumber(0.5, 0)
        with pytest.raises(ValueError):
            DualNumber(0, 3)**DualNumber(0.5, 5)

    def test_rpow(self):
        # power (non-dual, dual)
        x = (DualNumber(4,3))
        assert 2**x ==  DualNumber(2**4,3* math.log(2)*2**4)

    def test_rpow_zero(self):
        # power (non-dual == 0, dual)
        #
        # See test_pow_zero_dual.
        x = 0
        assert x**DualNumber(0.5, 0) == DualNumber(0, 0)
        assert x**DualNumber(0.5, 3) == DualNumber(0, 0)
        with pytest.raises(ValueError):
            x**DualNumber(-0.5, 0)
        with pytest.raises(ValueError):
            x**DualNumber(-0.5, 3)

    def test_neg(self):
        # neg (dual)
        x = DualNumber(1,2)
        assert -x == DualNumber(-1,-2)

    def test_floordiv(self):
        # floor-divide (dual, non-dual)
        x = DualNumber(1, 2)
        assert x // 3 == DualNumber(1//3, 2//3)

    def test_floordiv_dual(self):
        # floor-divide (dual, dual)
        x = DualNumber(1, 2)
        assert x // DualNumber(3,5) == DualNumber(1//3, (2*3 - 1*5)//(3*3) )

    def test_floordiv_zero(self):
        # floor-divide (dual, non-dual == 0)
        x = DualNumber(1, 2)
        with pytest.raises(ZeroDivisionError):
            x // 0

    def test_floordiv_zero_dual(self):
        # floor-divide (dual, dual with real part 0)
        x = DualNumber(1, 2)
        with pytest.raises(ZeroDivisionError):
            x // DualNumber(0,3)

    def test_rfloordiv_dual(self):
        # floor-divide (non-dual, dual)
        x = DualNumber(1, 2)
        assert 3 // x == DualNumber(3//1, (-3*2)//(1*1))

    def test_rfloordiv_zero_dual(self):
        # floor-divide (non-dual, dual with real part 0)
        x = DualNumber(0, 2)
        with pytest.raises(ZeroDivisionError):
            3 // x

    def test_eq(self):
        # equals (dual, non-dual)
        assert DualNumber(3,0) == 3
        assert not (DualNumber(3,0) == 4)
        assert not (DualNumber(3,1) == 3)

    def test_ne(self):
        # not-equals
        assert not (DualNumber(3,0) != 3)
        assert DualNumber(3,0) != 4
        assert DualNumber(3,1) != 3

    def test_ne_dual(self):
        assert DualNumber(1,2) != DualNumber(1,3)
        assert DualNumber(1,2) != DualNumber(3,2)
        assert not (DualNumber(1,2) != DualNumber(1,2))
