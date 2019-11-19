# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:26:50 2019
@author: sijie
"""

#import pytest
import ElementaryFunctions as EF
from DualNumber import DualNumber
import numpy as np


def test_sin():
    Test_Dual_Number_1 = DualNumber(1)
    assert EF.Sin(Test_Dual_Number_1).val == np.sin(1) and EF.Sin(Test_Dual_Number_1).der == np.cos(1)


def test_cos():
    Test_Dual_Number_1 = DualNumber(1)
    assert EF.Cos(Test_Dual_Number_1).val == np.cos(1) and EF.Cos(Test_Dual_Number_1).der == -np.sin(1)


def test_tan():
    Test_Dual_Number_1 = DualNumber(1)
    assert EF.Tan(Test_Dual_Number_1).val == np.tan(1) and EF.Tan(Test_Dual_Number_1).der == 1 / np.cos(1)**2


def test_exp():
    Test_Dual_Number_1 = DualNumber(1)
    assert EF.Exp(Test_Dual_Number_1).val == np.exp(1) and EF.Exp(Test_Dual_Number_1).der == np.exp(1)


def test_Power():
    Test_Dual_Number_2 = DualNumber(2)
    assert EF.Power(Test_Dual_Number_2, 2).val == 4 and EF.Power(Test_Dual_Number_2, 2).der == 4


def test_Log():
    Test_Dual_Number_1 = DualNumber(1)
    assert EF.Log(Test_Dual_Number_1).val == 0 and EF.Log(Test_Dual_Number_1).der == 1


def test_ArcSin():
    Test_Dual_Number_1 = DualNumber(0.5)
    assert EF.ArcSin(Test_Dual_Number_1).val == np.arcsin(0.5) and EF.ArcSin(Test_Dual_Number_1).der == 1 / np.sqrt(
        0.75)


def test_ArcCos():
    Test_Dual_Number_1 = DualNumber(0.5)
    assert EF.ArcCos(Test_Dual_Number_1).val == np.arccos(0.5) and EF.ArcCos(Test_Dual_Number_1).der == -1 / np.sqrt(
        0.75)


def test_ArcTan():
    Test_Dual_Number_1 = DualNumber(0.5)
    assert EF.ArcTan(Test_Dual_Number_1).val == np.arctan(0.5) and EF.ArcTan(Test_Dual_Number_1).der == 1 / np.sqrt(
        1.25)


def test_Sqrt():
    Test_Dual_Number_1 = DualNumber(0.5)
    assert EF.Sqrt(Test_Dual_Number_1).val == np.sqrt(0.5) and EF.Sqrt(Test_Dual_Number_1).der == 1 / np.sqrt(
        0.5) / 2
