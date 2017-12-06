#coding=utf-8
"""some short implementations of concepts from the lecture 'Mathematik für
Physiker 3'

TODO:
    Hesse-Matrix for arbitrary dimensions
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



def newton_kantorowitsch(func, X, x0, mode='iter', simple=False, tol=1e-9,
                         maxiter=50, _debug=False):
    """Calculate the zeros of func by numerical approximation using the
    Newton-Kantorowitsch-Method.

    Definition:
        standard: x_k+1 = x_k - Jacobian(func(x_k))**(-1) * func(x_k)
        simplified: x_k+1 = x_k - Jacobian(func(x_0))**(-1) * func(x_k)

    Parameters:
        func: IR^n -> IR^n, f_i(x_1, ..., x_n), given as list or Matrix
        X: set of x_i's in order, given as list or Matrix
        x0: starting point
        mode: recursive or iterative
        simple: use simplified methode
        tol: tolerance of solution
        maxiter: maximal iterations/recursions

    """
    # prepare and check input, create D_func_inv
    try:
        func = Matrix(func)
        X = Matrix(X)
        D_func_inv = func.jacobian(X).inv()
    except (AttributeError, TypeError):
        D_func_inv = (diff(func, X))**(-1)
    except NonSquareMatrixError:
        raise NonSquareMatrixError('func and X need to have same shape!')

    # create callable sequence according to the definition
    x0 = np.array(x0)
    arr2mat = [{'MatrixExpr': np.array}, 'numpy']
    seq1 = lambdify(X, X, modules=arr2mat)
    seq2 = lambdify(X, D_func_inv*func, modules=arr2mat)
    if simple:
        seq = lambda x: (seq1(*x) - seq2(*x0)).flatten()
    else:
        seq = lambda x: (seq1(*x) - seq2(*x)).flatten()

    # execution
    args = (seq, x0, tol, maxiter)
    if _debug:
        return seq1, seq2, args     # return arguments for inspection
    elif mode == 'iter':
        return _nk_iter(*args)
    elif mode == 'rec':
        return _nk_rec(*args)


def _nk_rec(seq, x_k, tol, maxiter):
    """newtons method, recursive"""
    if maxiter == 0 or np.all(x_k < tol):
        return x_k
    return _nk_rec(seq, seq(x_k), tol, maxiter-1)


def _nk_iter(seq, x_k, tol, maxiter):
    """newtons method, iterative"""
    i = 0
    while i < maxiter:
        if np.all(x_k < tol):
            return x_k
        x_k = seq(x_k)
        i += 1
    return x_k
