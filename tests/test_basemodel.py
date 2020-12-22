"""
Test `basemodel.py`
"""
import pytest
from neural.basemodel import _dict_add_, _dict_add_scalar_, _dict_iadd_


@pytest.fixture(scope="module")
def dict_vars() -> tuple:
    a = {"a": 0, "b": 1, "c": "Hello"}
    b = {"a": 1, "b": -1, "c": " World"}
    c = 5
    ab_ref = {"a": 1, "b": 0, "c": "Hello World"}
    abc_ref = {"a": 5, "b": -4, "c": "Hello" + c * " World"}
    return (a, b, c, ab_ref, abc_ref)


def test_dict_utils(dict_vars):
    a, b, c, ab_ref, abc_ref = dict_vars
    assert _dict_add_(a, b) == ab_ref
    assert _dict_add_scalar_(a, b, c) == abc_ref
    tmp = _dict_iadd_(a, b)
    assert tmp == ab_ref
    assert id(tmp) == id(a)
