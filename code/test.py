import DualNumber
from ElementaryFunctions import *
import pytest

def test_overload_add():
    x = DualNumber(5)
    y = DualNumber(7)

    assert (x+y).val==12 and (x+y).der==2

def test_overload_add_types():
    with pytest.raises(AssertionError):
        x = DualNumber(5)
        x+"test"



