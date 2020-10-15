import numpy as np
from src.inverse import invert_matrix, reduced_row_echelon_form
import pytest

def test_simple_matrix():
    x = np.array([[1, 2], [3, 4]])
    ans = invert_matrix(x.copy())

    x_inv = np.array([[-2, 1], [3 / 2, -1 / 2]])
    assert (x_inv == ans).all()

def test_non_invertable():
    x = np.array([[1,0,0],[0,1,0],[0,0,0]])
    with pytest.raises(ValueError) as e:
        invert_matrix(x)
    assert str(e.value) == 'Non singular matrix'


def test_rref():
    x = np.asarray([[1., 2., -1., -4.],
                    [2., 3., -1., -11.],
                    [-2, 0., -3.,  22]])

    ans = reduced_row_echelon_form(x.copy(), *x.shape)
    y = np.asarray([[1,0,0,-8],[0,1,0,1],[0,0,1,-2]])
    assert (y == ans).all()


