#!/anaconda3/bin/python
# -*- coding: utf-8 -*-

from DualNumber_with_Reverse import DualNumber
import ElementaryFunctions_with_reverse as EF
import numpy as np
from scipy.special import logit
import pytest

# DualNumber tests
def test_overload_add():
    x = DualNumber(5,1)
    y = DualNumber(7,1)

    assert (x + y).val == 12 and (x + y).der == 2
    assert (x + 5).val == 10 and (x + 5).der == 1
    assert (x + 5.0).val == 10 and (x + 5.0).der == 1
    assert (5 + x).val == 10 and (5 + x).der == 1
    assert (5.0 + x).val == 10 and (5.0 + x).der == 1

def test_overload_add_types():
    with pytest.raises(AssertionError):
        x = DualNumber(5)
        x+"test"

def test_overload_multiply():
    x = DualNumber(5)
    y = DualNumber(7,2)

    assert (x * y).val == 35 and (x * y).der == 17
    assert (x * 5).val == 25 and (x * 5).der == 5
    assert (x * 5.0).val == 25 and (x * 5.0).der == 5
    assert (5.0 * x).val == 25 and (5.0 * x).der == 5
    assert (5 * x).val == 25 and (5 * x).der == 5

def test_overload_multiply_types():
    with pytest.raises(AssertionError):
        x = DualNumber(5)
        x*"test"

def test_overload_sub():
    x = DualNumber(5,2)
    y = DualNumber(7,1)

    assert (x - y).val == -2 and (x - y).der == 1
    assert (x.__rsub__(y)).val == 2 and (x.__rsub__(y)).der == -1
    assert (x - 5).val == 0 and (x - 5).der == 2
    assert (x - 5.0).val == 0 and (x - 5.0).der == 2
    assert (5 - x).val == 0 and (5 - x).der == -2
    assert (5.0 - x).val == 0 and (5.0 - x).der == -2

def test_overload_sub_types():
    with pytest.raises(AssertionError):
        x = DualNumber(5)
        x - "test"

def test_overload_truediv():
    x = DualNumber(10,2)
    y = DualNumber(2,4)
    print((x / y).der)
    assert (x / y).val == 5 and (x / y).der == (2*2-10*4)/(2**2)
    assert (x.__rtruediv__(y)).val == 0.2 and (x.__rtruediv__(y)).der == (4*10-2*2)/(10*10)
    assert (x / 5).val == 2 and (x / 5).der == 0.4
    assert (x / 5.0).val == 2.0 and (x / 5.0).der == 0.4
    assert (5 / x).val == 0.5 and (5 / x).der == -0.1
    assert (5.0 / x).val == 0.5 and (5.0 / x).der == -0.1

def test_overload_truediv_types():
    with pytest.raises(AssertionError):
      x = DualNumber(10)
      x/"test"

def test_overload_power():
    x = DualNumber(2,2)
    y = DualNumber(3,4)

    assert (x**y).val == 8 and (x**y).der == 8*(3/2*2+4*np.log(2))
    assert (x.__rpow__(y)).val == 9 and (x.__rpow__(y)).der == 9*(2/3*4+2*np.log(3))
    assert (x**2).val == 4 and (x**2).der == 8
    assert (x**2.0).val == 4 and (x**2.0).der == 8
    assert (2**x).val == 4 and (2**x).der == np.log(2)*4
    assert (2.0**x).val == 4 and (2.0**x).der == np.log(2)*4

def test_overload_power_types():
    with pytest.raises(AssertionError):
        x = DualNumber(5)
        x**"test"

def test_overload_pos():
    x = DualNumber(5,1)
    y = +x
    assert y.val == 5 and y.der == 1

def test_overload_neg():
    x = DualNumber(5,1)
    y = -x
    assert y.val == -5 and y.der == -1

def test_overload_abs():
    x = DualNumber(-5,-1)
    y = abs(x)
    assert y.val == 5 and y.der == 1
    x = DualNumber(5,1)
    y = abs(x)
    assert y.val == 5 and y.der == 1

def test_overload_round():
    x = DualNumber(5.123,1.0012)
    y = round(x,2)
    assert y.val == 5.12 and y.der == 1.00
    
def test_repr():
    x = DualNumber(5,1)
    assert repr(x) == 'Derivative: 1.00' + '\n' + 'Value: 5.00'
    
def test_str():
    x = DualNumber(5,1)
    assert print(x) == print('Derivative: 1.00' + '\n' + 'Value: 5.00')
    
def test_set_der():
    x = DualNumber(5,1)
    x.set_der(2)
    assert x.der == 2

def test_set_val():
    x = DualNumber(5,1)
    x.set_val(2)
    assert x.val == 2
    
def test_jacobian():
    x = DualNumber(5,1)
    assert x.jacobian == 1

def test_overload_eq():
    x = DualNumber(5,1)
    y = DualNumber(5,1)
    assert x == y

def test_overload_ne():
    x = DualNumber(5,1)
    y = DualNumber(6,1)
    z = DualNumber(5,2)
    assert x != y
    assert x != z
    
test_overload_add()
test_overload_add_types()
test_overload_multiply()
test_overload_multiply_types()
test_overload_sub()
test_overload_sub_types()
test_overload_truediv()
test_overload_truediv_types()
test_overload_pos()
test_overload_neg()
test_overload_abs()
test_overload_round()
test_repr()
test_str()
test_set_der()
test_set_val()
test_jacobian()
test_overload_eq()
test_overload_ne()

# Elementary Function tests
def test_sin():
    Test_Dual_Number_1 = DualNumber(1)
    assert EF.Sin(Test_Dual_Number_1).val == np.sin(1) and EF.Sin(Test_Dual_Number_1).der == np.cos(1)
    Test_Dual_Number_1 = 1
    assert EF.Sin(Test_Dual_Number_1).val == np.sin(1) and EF.Sin(Test_Dual_Number_1).der == 0

def test_cos():
    Test_Dual_Number_1 = DualNumber(1)
    assert EF.Cos(Test_Dual_Number_1).val == np.cos(1) and EF.Cos(Test_Dual_Number_1).der == -np.sin(1)
    Test_Dual_Number_1 = 1
    assert EF.Cos(Test_Dual_Number_1).val == np.cos(1) and EF.Cos(Test_Dual_Number_1).der == 0

def test_tan():
    Test_Dual_Number_1 = DualNumber(1)
    assert EF.Tan(Test_Dual_Number_1).val == np.tan(1) and EF.Tan(Test_Dual_Number_1).der == 1 / np.cos(1)**2
    Test_Dual_Number_1 = 1
    assert EF.Tan(Test_Dual_Number_1).val == np.tan(1) and EF.Tan(Test_Dual_Number_1).der == 0

def test_exp():
    Test_Dual_Number_1 = DualNumber(1)
    assert EF.Exp(Test_Dual_Number_1).val == np.exp(1) and EF.Exp(Test_Dual_Number_1).der == np.exp(1)
    Test_Dual_Number_1 = 1
    assert EF.Exp(Test_Dual_Number_1).val == np.exp(1) and EF.Exp(Test_Dual_Number_1).der == 0

def test_Power():
    Test_Dual_Number_2 = DualNumber(2)
    assert EF.Power(Test_Dual_Number_2, 2).val == 4 and EF.Power(Test_Dual_Number_2, 2).der == 4
    Test_Dual_Number_2 = 2
    assert EF.Power(Test_Dual_Number_2, 2).val == 4 and EF.Power(Test_Dual_Number_2, 2).der == 0

def test_Log():
    Test_Dual_Number_1 = DualNumber(1)
    assert EF.Log(Test_Dual_Number_1).val == 0 and EF.Log(Test_Dual_Number_1).der == 1
    Test_Dual_Number_1 = 1
    assert EF.Log(Test_Dual_Number_1).val == 0 and EF.Log(Test_Dual_Number_1).der == 0
    
    Test_Dual_Number_1 = DualNumber(2)
    assert EF.Log(Test_Dual_Number_1,base=10).val == np.log(2) / np.log(10)  and EF.Log(Test_Dual_Number_1,base=10).der == 1/(np.log(10)*2)
    Test_Dual_Number_1 = 2
    assert EF.Log(Test_Dual_Number_1,base=10).val == np.log(2) / np.log(10) and EF.Log(Test_Dual_Number_1,base=10).der == 0

def test_ArcSin():
    Test_Dual_Number_1 = DualNumber(0.5)
    assert EF.ArcSin(Test_Dual_Number_1).val == np.arcsin(0.5) and EF.ArcSin(Test_Dual_Number_1).der == 1 / np.sqrt(
        0.75)
    Test_Dual_Number_1 = 0.5
    assert EF.ArcSin(Test_Dual_Number_1).val == np.arcsin(0.5) and EF.ArcSin(Test_Dual_Number_1).der == 0

def test_ArcCos():
    Test_Dual_Number_1 = DualNumber(0.5)
    assert EF.ArcCos(Test_Dual_Number_1).val == np.arccos(0.5) and EF.ArcCos(Test_Dual_Number_1).der == -1 / np.sqrt(
        0.75)
    Test_Dual_Number_1 = 0.5
    assert EF.ArcCos(Test_Dual_Number_1).val == np.arccos(0.5) and EF.ArcCos(Test_Dual_Number_1).der == 0

def test_ArcTan():
    Test_Dual_Number_1 = DualNumber(0.5)
    assert EF.ArcTan(Test_Dual_Number_1).val == np.arctan(0.5) and EF.ArcTan(Test_Dual_Number_1).der == 1 / 1.25
    Test_Dual_Number_1 = 0.5
    assert EF.ArcTan(Test_Dual_Number_1).val == np.arctan(0.5) and EF.ArcTan(Test_Dual_Number_1).der == 0

def test_Sqrt():
    Test_Dual_Number_1 = DualNumber(0.5)
    assert EF.Sqrt(Test_Dual_Number_1).val == np.sqrt(0.5) and EF.Sqrt(Test_Dual_Number_1).der == 1 / np.sqrt(
        0.5) / 2
    Test_Dual_Number_1 = 0.5
    assert EF.Sqrt(Test_Dual_Number_1).val == np.sqrt(0.5) and EF.Sqrt(Test_Dual_Number_1).der == 0

def test_data_type_check():
    x = 'x'
    try:
        EF.data_type_check(x)
    except AttributeError:
        return 0

def test_sinh():
    Test_Dual_Number_1 = DualNumber(1)
    assert EF.Sinh(Test_Dual_Number_1).val == np.sinh(1) and EF.Sinh(Test_Dual_Number_1).der == np.cosh(1)
    Test_Dual_Number_1 = 1
    assert EF.Sinh(Test_Dual_Number_1).val == np.sinh(1) and EF.Sin(Test_Dual_Number_1).der == 0

def test_cosh():
    Test_Dual_Number_1 = DualNumber(1)
    assert EF.Cosh(Test_Dual_Number_1).val == np.cosh(1) and EF.Cosh(Test_Dual_Number_1).der == np.sinh(1)
    Test_Dual_Number_1 = 1
    assert EF.Cosh(Test_Dual_Number_1).val == np.cosh(1) and EF.Cosh(Test_Dual_Number_1).der == 0

def test_tanh():
    Test_Dual_Number_1 = DualNumber(1)
    assert EF.Tanh(Test_Dual_Number_1).val == np.tanh(1) and EF.Tanh(Test_Dual_Number_1).der == 1 / np.cosh(1)**2
    Test_Dual_Number_1 = 1
    assert EF.Tanh(Test_Dual_Number_1).val == np.tanh(1) and EF.Tanh(Test_Dual_Number_1).der == 0

def test_logit():
    Test_Dual_Number_1 = DualNumber(1)
    assert EF.Logit(Test_Dual_Number_1).val == logit(1) and EF.Logit(Test_Dual_Number_1).der == np.exp(-1) / (1+np.exp(-1))**2
    Test_Dual_Number_1 = 1
    assert EF.Logit(Test_Dual_Number_1).val == logit(1) and EF.Logit(Test_Dual_Number_1).der == 0

    
test_sin()      
test_cos()
test_tan()
test_exp()
test_Power()
test_Log()
test_ArcSin()
test_ArcCos()
test_ArcTan()
test_Sqrt()
test_data_type_check()
test_sinh()      
test_cosh()
test_tanh()
test_logit()

# Reverse Mode tests
def rm_test_overload_add():
    x = DualNumber(5,Reverse=True)
    y = DualNumber(7,Reverse=True)

    assert (x + y).val == 12 and (x + y).der == 0
    assert (x + 5).val == 10 and (x + 5).der == 0
    assert (x + 5.0).val == 10 and (x + 5.0).der == 0
    assert (5 + x).val == 10 and (5 + x).der == 0
    assert (5.0 + x).val == 10 and (5.0 + x).der == 0

def rm_test_overload_add_types():
    with pytest.raises(AssertionError):
        x = DualNumber(5,Reverse=True)
        x+"test"

def rm_test_overload_multiply():
    x = DualNumber(5,Reverse=True)
    y = DualNumber(7,Reverse=True)

    assert (x * y).val == 35 and (x * y).der == 0
    assert (x * 5).val == 25 and (x * 5).der == 0
    assert (x * 5.0).val == 25 and (x * 5.0).der == 0
    assert (5.0 * x).val == 25 and (5.0 * x).der == 0
    assert (5 * x).val == 25 and (5 * x).der == 0

def rm_test_overload_multiply_types():
    with pytest.raises(AssertionError):
        x = DualNumber(5,Reverse=True)
        x*"test"

def rm_test_overload_sub():
    x = DualNumber(5,Reverse=True)
    y = DualNumber(7,Reverse=True)

    assert (x - y).val == -2 and (x - y).der == 0
    assert (x.__rsub__(y)).val == 2 and (x.__rsub__(y)).der == 0
    assert (x - 5).val == 0 and (x - 5).der == 0
    assert (x - 5.0).val == 0 and (x - 5.0).der == 0
    assert (5 - x).val == 0 and (5 - x).der == 0
    assert (5.0 - x).val == 0 and (5.0 - x).der == 0

def rm_test_overload_sub_types():
    with pytest.raises(AssertionError):
        x = DualNumber(5,Reverse=True)
        x - "test"

def rm_test_overload_truediv():
    x = DualNumber(10,Reverse=True)
    y = DualNumber(2,Reverse=True)
    print((x / y).der)
    assert (x / y).val == 5 and (x / y).der == 0
    assert (x.__rtruediv__(y)).val == 0.2 and (x.__rtruediv__(y)).der == 0
    assert (x / 5).val == 2 and (x / 5).der == 0
    assert (x / 5.0).val == 2.0 and (x / 5.0).der == 0
    assert (5 / x).val == 0.5 and (5 / x).der == 0
    assert (5.0 / x).val == 0.5 and (5.0 / x).der == 0

def rm_test_overload_truediv_types():
    with pytest.raises(AssertionError):
      x = DualNumber(10,Reverse=True)
      x/"test"

def rm_test_overload_power():
    x = DualNumber(2,Reverse=True)
    y = DualNumber(3,Reverse=True)

    assert (x**y).val == 8 and (x**y).der == 0
    assert (x.__rpow__(y)).val == 9 and (x.__rpow__(y)).der == 0
    assert (x**2).val == 4 and (x**2).der == 0
    assert (x**2.0).val == 4 and (x**2.0).der == 0
    assert (2**x).val == 4 and (2**x).der == 0
    assert (2.0**x).val == 4 and (2.0**x).der == 0

def rm_test_overload_power_types():
    with pytest.raises(AssertionError):
        x = DualNumber(5,Reverse=True)
        x**"test"

def rm_test_overload_pos():
    x = DualNumber(5,Reverse=True)
    y = +x
    assert y.val == 5 and y.der == 0

def rm_test_overload_neg():
    x = DualNumber(5,Reverse=True)
    y = -x
    assert y.val == -5 and y.der == 0

def rm_test_overload_abs():
    x = DualNumber(-5,Reverse=True)
    y = abs(x)
    assert y.val == 5 and y.der == 0
    x = DualNumber(5,1,Reverse=True)
    y = abs(x)
    assert y.val == 5 and y.der == 0

def rm_test_overload_round():
    x = DualNumber(5.123,Reverse=True)
    y = round(x,2)
    assert y.val == 5.12 and y.der == 0
    
def rm_test_repr():
    x = DualNumber(5,Reverse=True)
    assert repr(x) == 'Derivative: 0.00' + '\n' + 'Value: 5.00'
    
def rm_test_str():
    x = DualNumber(5,Reverse=True)
    assert print(x) == print('Derivative: 0.00' + '\n' + 'Value: 5.00')
    
def rm_test_jacobian():
    x = DualNumber(5,Reverse=True)
    assert x.jacobian == 0

def rm_test_overload_eq():
    x = DualNumber(5,Reverse=True)
    y = DualNumber(5,Reverse=True)
    assert x == y

def rm_test_overload_ne():
    x = DualNumber(5,1,Reverse=True)
    y = DualNumber(6,1,Reverse=True)
    z = DualNumber(5,2,Reverse=True)
    assert x != y
    assert x != z

def rm_test_sin():
    Test_Dual_Number_1 = DualNumber(1,Reverse=True)
    assert EF.Sin(Test_Dual_Number_1).val == np.sin(1) and EF.Sin(Test_Dual_Number_1).der == np.cos(1)
    Test_Dual_Number_1 = 1
    assert EF.Sin(Test_Dual_Number_1).val == np.sin(1) and EF.Sin(Test_Dual_Number_1).der == 0

def rm_test_cos():
    Test_Dual_Number_1 = DualNumber(1,Reverse=True)
    assert EF.Cos(Test_Dual_Number_1).val == np.cos(1) and EF.Cos(Test_Dual_Number_1).der == -np.sin(1)
    Test_Dual_Number_1 = 1
    assert EF.Cos(Test_Dual_Number_1).val == np.cos(1) and EF.Cos(Test_Dual_Number_1).der == 0

def rm_test_tan():
    Test_Dual_Number_1 = DualNumber(1,Reverse=True)
    assert EF.Tan(Test_Dual_Number_1).val == np.tan(1) and EF.Tan(Test_Dual_Number_1).der == 1 / np.cos(1)**2
    Test_Dual_Number_1 = 1
    assert EF.Tan(Test_Dual_Number_1).val == np.tan(1) and EF.Tan(Test_Dual_Number_1).der == 0

def rm_test_exp():
    Test_Dual_Number_1 = DualNumber(1,Reverse=True)
    assert EF.Exp(Test_Dual_Number_1).val == np.exp(1) and EF.Exp(Test_Dual_Number_1).der == np.exp(1)
    Test_Dual_Number_1 = 1
    assert EF.Exp(Test_Dual_Number_1).val == np.exp(1) and EF.Exp(Test_Dual_Number_1).der == 0

def rm_test_Power():
    Test_Dual_Number_2 = DualNumber(2,Reverse=True)
    assert EF.Power(Test_Dual_Number_2, 2).val == 4 and EF.Power(Test_Dual_Number_2, 2).der == 4
    Test_Dual_Number_2 = 2
    assert EF.Power(Test_Dual_Number_2, 2).val == 4 and EF.Power(Test_Dual_Number_2, 2).der == 0

def rm_test_Log():
    Test_Dual_Number_1 = DualNumber(1,Reverse=True)
    assert EF.Log(Test_Dual_Number_1).val == 0 and EF.Log(Test_Dual_Number_1).der == 1
    Test_Dual_Number_1 = 1
    assert EF.Log(Test_Dual_Number_1).val == 0 and EF.Log(Test_Dual_Number_1).der == 0
    
    Test_Dual_Number_1 = DualNumber(2,Reverse=True)
    assert EF.Log(Test_Dual_Number_1,base=10).val == np.log(2) / np.log(10)  and EF.Log(Test_Dual_Number_1,base=10).der == 1/(np.log(10)*2)
    Test_Dual_Number_1 = 2
    assert EF.Log(Test_Dual_Number_1,base=10).val == np.log(2) / np.log(10) and EF.Log(Test_Dual_Number_1,base=10).der == 0

def rm_test_ArcSin():
    Test_Dual_Number_1 = DualNumber(0.5,Reverse=True)
    assert EF.ArcSin(Test_Dual_Number_1).val == np.arcsin(0.5) and EF.ArcSin(Test_Dual_Number_1).der == 1 / np.sqrt(
        0.75)
    Test_Dual_Number_1 = 0.5
    assert EF.ArcSin(Test_Dual_Number_1).val == np.arcsin(0.5) and EF.ArcSin(Test_Dual_Number_1).der == 0

def rm_test_ArcCos():
    Test_Dual_Number_1 = DualNumber(0.5,Reverse=True)
    assert EF.ArcCos(Test_Dual_Number_1).val == np.arccos(0.5) and EF.ArcCos(Test_Dual_Number_1).der == -1 / np.sqrt(
        0.75)
    Test_Dual_Number_1 = 0.5
    assert EF.ArcCos(Test_Dual_Number_1).val == np.arccos(0.5) and EF.ArcCos(Test_Dual_Number_1).der == 0

def rm_test_ArcTan():
    Test_Dual_Number_1 = DualNumber(0.5,Reverse=True)
    assert EF.ArcTan(Test_Dual_Number_1).val == np.arctan(0.5) and EF.ArcTan(Test_Dual_Number_1).der == 1 / 1.25
    Test_Dual_Number_1 = 0.5
    assert EF.ArcTan(Test_Dual_Number_1).val == np.arctan(0.5) and EF.ArcTan(Test_Dual_Number_1).der == 0

def rm_test_Sqrt():
    Test_Dual_Number_1 = DualNumber(0.5,Reverse=True)
    assert EF.Sqrt(Test_Dual_Number_1).val == np.sqrt(0.5) and EF.Sqrt(Test_Dual_Number_1).der == 1 / np.sqrt(
        0.5) / 2
    Test_Dual_Number_1 = 0.5
    assert EF.Sqrt(Test_Dual_Number_1).val == np.sqrt(0.5) and EF.Sqrt(Test_Dual_Number_1).der == 0

def rm_test_sinh():
    Test_Dual_Number_1 = DualNumber(1,Reverse=True)
    assert EF.Sinh(Test_Dual_Number_1).val == np.sinh(1) and EF.Sinh(Test_Dual_Number_1).der == np.cosh(1)
    Test_Dual_Number_1 = 1
    assert EF.Sinh(Test_Dual_Number_1).val == np.sinh(1) and EF.Sin(Test_Dual_Number_1).der == 0

def rm_test_cosh():
    Test_Dual_Number_1 = DualNumber(1,Reverse=True)
    assert EF.Cosh(Test_Dual_Number_1).val == np.cosh(1) and EF.Cosh(Test_Dual_Number_1).der == np.sinh(1)
    Test_Dual_Number_1 = 1
    assert EF.Cosh(Test_Dual_Number_1).val == np.cosh(1) and EF.Cosh(Test_Dual_Number_1).der == 0

def rm_test_tanh():
    Test_Dual_Number_1 = DualNumber(1,Reverse=True)
    assert EF.Tanh(Test_Dual_Number_1).val == np.tanh(1) and EF.Tanh(Test_Dual_Number_1).der == 1 / np.cosh(1)**2
    Test_Dual_Number_1 = 1
    assert EF.Tanh(Test_Dual_Number_1).val == np.tanh(1) and EF.Tanh(Test_Dual_Number_1).der == 0

def rm_test_logit():
    Test_Dual_Number_1 = DualNumber(1,Reverse=True)
    assert EF.Logit(Test_Dual_Number_1).val == logit(1) and EF.Logit(Test_Dual_Number_1).der == np.exp(-1) / (1+np.exp(-1))**2
    Test_Dual_Number_1 = 1
    assert EF.Logit(Test_Dual_Number_1).val == logit(1) and EF.Logit(Test_Dual_Number_1).der == 0

rm_test_overload_add()
rm_test_overload_add_types()
rm_test_overload_multiply()
rm_test_overload_multiply_types()
rm_test_overload_sub()
rm_test_overload_sub_types()
rm_test_overload_truediv()
rm_test_overload_truediv_types()
rm_test_overload_pos()
rm_test_overload_neg()
rm_test_overload_abs()
rm_test_overload_round()
rm_test_repr()
rm_test_str()
rm_test_jacobian()
rm_test_overload_eq()
rm_test_overload_ne()
    
rm_test_sin()      
rm_test_cos()
rm_test_tan()
rm_test_exp()
rm_test_Power()
rm_test_Log()
rm_test_ArcSin()
rm_test_ArcCos()
rm_test_ArcTan()
rm_test_Sqrt()
rm_test_sinh()      
rm_test_cosh()
rm_test_tanh()
rm_test_logit()


