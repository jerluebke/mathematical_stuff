#coding=utf-8
"""some short implementations of concepts from the lecture 'Mathematik für
Physiker 3'

TODO:
    Gradient, Jacobi-Matrix, Hesse-Matrix for arbitrary dimensions
    Taylor
"""

from sympy import *
import numpy as np


def pd2(func, x, y):
    """differentiate partially func regarding x and y"""
    return diff(diff(func, x), y).simplify()


def grad(func):
    """return gradient of func, func: IR^n -> IR"""
    return Matrix([[diff(f, x_j) for x_j in func.free_symbols]])


def jacobi(func):
    """return Jacobian of func, func: IR^n -> IR^m
    If func is not iterabel, i.e. func: IR^n -> IR, return its gradient

    TODO:
    add free symbols as list as parameter"""
    try:
        return Matrix([[diff(f_i, x_j) for x_j in func.free_symbols] for f_i in func])
    except TypeError:
        return grad(func)


def hesse2(func):
    """return Hesse-Matrix (all second partial derivatives) of func
    func: IR^2 -> IR"""
    x, y = func.free_symbols
    return Matrix([[pd2(func, x, x), pd2(func, x, y)],
                   [pd2(func, y, x), pd2(func, y, y)]])



def newton_kantorowitsch_rec(func, X, x0, tol, maxrec):
    """Calculate the zeros of func by numerical approximation using the
    Newton-Kantorowitsch-Method.
    Calculation is done recursive.

    Definition:
        x_k+1 = x_k - Jacobian(func(x_k))**(-1) * func(x_k)

    Parameters:
        func: IR^n -> IR^n, f_i(x_1, ..., x_n), given as list or Matrix
        X: set of x_i's in order, given as list or Matrix
        x0: starting point for recursion
        tol: tolerance of solution
        maxrec: maximal recursion depth

    see also:
        newton_kantorowitsch_iter
        simple_newton_rec
        simple_newton_iter

    """


def newton_kantorowitsch_iter(func, X, x0, tol, maxiter):
    """see newton_kantorowitsch_rec, but calculation is done iterative"""


def simple_newton_rec(func, X, x0, tol, maxrec):
    """simplified version of Newton-Kantorowitsch-Method, for reference see
    newton_kantorowitsch_rec.
    Calculation is done recursive.

    Definition:
        TODO

    """


def simple_newton_iter(func, X, x0, tol, maxiter):
    """see simple_newton_rec, but calculation is done iterative"""
