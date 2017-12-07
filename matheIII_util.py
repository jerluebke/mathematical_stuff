#coding=utf-8
"""some short implementations of concepts from the lecture 'Mathematik für
Physiker 3'

TODO:
    Hesse-Matrix for arbitrary dimensions
    Taylor
"""

from sympy import *
import numpy as np


# initialization
x, y, z, t = symbols('x y z t')
x_1, x_2, x_3 = symbols('x_1, x_2, x_3')
f, g, h = symbols('f g h', cls=Function)

init_printing()


def pd2(func, x, y):
    """differentiate partially func regarding x and y"""
    return diff(diff(func, x), y).simplify()


def grad(func):
    """return gradient of func, func: IR^n -> IR

    TODO:
    add free symbols as list as parameter"""
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



def newton_kantorowitsch(func, X, x0, simple=False, tol=1e-9,
                         maxiter=50, printall=False, test=False, _debug=False):
    """Calculate the zeros of func by numerical approximation using the
    Newton-Kantorowitsch-Method.

    Definition:
        standard: x_k+1 = x_k - Jacobian(func(x_k))**(-1) * func(x_k)
        simplified: x_k+1 = x_k - Jacobian(func(x_0))**(-1) * func(x_k)

    Parameters:
        func: IR^n -> IR^n, f_i(x_1, ..., x_n), given as list or Matrix
        X: set of x_i's in order, given as list or Matrix
        x0: starting point
        simple: use simplified methode
        tol: tolerance of solution
        maxiter: maximal iterations/recursions
        printall: verbose execution
        test: test during execution and return if x_k < tol, test final result

    """
    # prepare and check input, create D_func_inv
    try:
        x0 = [float(x0_i) for x0_i in x0]
    except TypeError:
        raise TypeError('x0 has to be a tuple or list of numbers!')
    try:
        func = Matrix(func)
        X = Matrix(X)
        D_func_inv = func.jacobian(X).inv()
    except (AttributeError, TypeError):
        X = X[0]
        D_func_inv = (diff(func, X))**(-1)
    except NonSquareMatrixError:
        raise NonSquareMatrixError('func and X need to have same shape!')

    # create callable sequence according to the definition
    x0 = np.array(x0)
    mat2arr = [{'MatrixExpr': np.array}, 'numpy']
    if simple:
        # Df(x)**(-1) = Df(x0)**(-1)
        D_func_inv = _subs(D_func_inv, X, x0)

    # x_k+1 = x_k - Df(x_k)**(-1) * f(x_k)
    _seq = lambdify(X, X-D_func_inv*func, modules=mat2arr)
    _func = lambda x: (lambdify(X, func, modules=mat2arr))(*x).flatten()
    seq = lambda x: (_seq(*x)).flatten()

    if printall:
        # print('\n'+pretty(X), '-', pretty(D_func_inv), '*', pretty(func)+'\n')
        print('\nFunction:\n\n', pretty(func))
        print('\nInverse Jacobian:\n\n', pretty(D_func_inv), '\n\n')

    # execution
    args = (seq, x0, _func, tol, maxiter, printall, test)
    if _debug:
        return args         # return arguments for inspection
    res = _nk_iter(*args)
    if printall and test:
        print('\nTest: f(x_res) = '+pretty(_func(res))+'\n')
    return res


nk = newton_kantorowitsch


def _nk_iter(seq, x_k, _func, tol, maxiter, printall, test):
    """newtons method, iterative"""
    i = 0
    while i < maxiter:
        if test and np.all(np.abs(_func(x_k)) < tol):
            return x_k
        i += 1
        if printall:
            print(i, x_k)
        x_k = seq(x_k)
    return x_k


def _subs(expr, X, X_subs):
    """helper function for multiple substitutions
    expr is the expression whose variables are to be substituted
    X is a list or Matrix with the regarded variables in order
    X_subs are the new values in order
    """
    i = 0
    while i < len(X):
        expr = expr.subs(X[i], X_subs[i])
        i += 1
    return expr



def example(which=0):
    """create some example data
    TODO: expand"""
    if which == 0:
        f_sym = (x**3 - 3*x*y**2 - 1, 3*x**2*y - y**3)
        f_lam = lambdify((x, y), f_sym)
    return f_sym, f_lam
