#coding=utf-8
"""some short implementations of concepts from the lecture 'Mathematik für
Physiker 3'

TODO:
    Taylor
"""

from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp


# initialization
x, y, z, t = symbols('x y z t')
x_1, x_2, x_3 = symbols('x_1, x_2, x_3')
f, g, h = symbols('f g h', cls=Function)

init_printing()


def pd2(func, x, y):
    """differentiate partially func regarding x and y"""
    return diff(diff(func, x), y).simplify()



def newton_kantorowitsch(func, X, x0, simple=False, tol=1e-9,
                         maxiter=50, printall=False, _debug=False):
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
        tol: tolerance for convergance of x_k
        maxiter: maximal iterations/recursions
        printall: verbose execution

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
        x0 = x0[0]
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

    # Sequence: x_k+1 = x_k - Df(x_k)**(-1) * f(x_k)
    _seq = lambdify(X, X-D_func_inv*func, modules=mat2arr)
    seq = lambda x: (_seq(*x)).flatten()

    if printall:
        print('\nFunction:\n\n', pretty(func))
        print('\nInverse Jacobian:\n\n', pretty(D_func_inv), '\n\n')

    # execution
    args = (seq, x0, tol, maxiter, printall)
    if _debug:
        return args         # return arguments for inspection
    res = _nk_iter(*args)
    if printall:
        _func = lambda x: (lambdify(X, func, modules=mat2arr))(*x).flatten()
        print('\nTest: f(x_res) = '+pretty(_func(res))+'\n')
    return res


nk = newton_kantorowitsch


def _nk_iter(seq, x_k, tol, maxiter, printall):
    """newtons method, iterative"""
    i = 0
    while i < maxiter:
        i += 1
        if printall:
            print(i, x_k)
        x_next = seq(x_k)
        if np.all(np.abs(x_k - x_next) < tol):
            return x_k
        x_k = x_next
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



def plot_phase_trajectories(func, inits, xbound, ybound, tbound=(0, 10, 100),
                            use_sage=False, axis=None,
                            odeint_kwargs={}, plt_kwargs={'color': 'blue'}):
    """function for plotting the  phase space trajectories of ordinary
    differential equations (ODEs) of the form dy/dt = f(y, t), where y can be
    a n-dim vector (see scipy.integrate.odeint for reference).
    Inspired by plotdf; reference: github.com/jmoy

    Parameters:
        func - RHS of ODE; callable(t, y)
        inits - initial values for plotting trajectories
        xbound, ybound - sequences with len = 2 gives min and max values for x
            and y respectivly
        tbound - sequence with len = 3 gives min, max and step for t values to
            calculate the trajectories for
        use_sage - whether to use sage for plotting instead of matplotlib
        axis - matplotlib axis to draw plot in; if None, current will be used
        odeint_kwargs - dict containing kwargs for scipy.integrate.odeint
        plt_kwargs - dict containing kwargs for matplotlib.pyplot; can also be
            used for sage-Graphics objects

    Returns:
        list containing matplotlib-artist objects (Line2D)
        or one sage-Graphics object
    """
    f = lambda x, t: func(t, x)
    if use_sage:
        from sage.plot.graphics import Graphics
        from sage.plot.line import line

    elif axis is None:
        axis = plt.gca()

    def f_neg(x, t):
        return -f(x, t)

    artists = [] if not use_sage else Graphics()
    t = np.linspace(*tbound)
    for i in inits:
        sol_fwd = odeint(f, i, t, **odeint_kwargs)          # forward solution
        sol_bwd = odeint(f_neg, i, t, **odeint_kwargs)      # backward solution
        sol = np.vstack((np.flipud(sol_bwd), sol_fwd))      # flip sol_bwd and put both together
        sol_x = sol[:,0]                                    # left column of sol
        sol_y = sol[:,1]                                    # right column of sol
        sol_x_masked = np.ma.masked_outside(sol_x, *xbound) # mask data to prevent
        sol_y_masked = np.ma.masked_outside(sol_y, *ybound) #  blow-up of solution
        if not use_sage:
            artists.append(axis.plot(sol_x_masked, sol_y_masked,
                                     **plt_kwargs))
        else:
            artists += line(zip(sol_x_masked, sol_y_masked), plt_kwargs,
                            xmin=xbound[0], xmax=xbound[1])

    if not use_sage:
        plt.xlim(xbound)
        plt.ylim(ybound)

    return artists


ppt = plot_phase_trajectories


def ppt_solve_ivp(f, inits, xbound, ybound, t=(0, 10), steps=100,
                  axis=None, sivp_kwargs={}, plt_kwargs={'c': 'b'}):
    """Plots phase trajectory of given ODE with scipy.integrate.solve_ivp
    For reference see ´plot_phase_trajectories´
    Input is mostly the same, apart from t, which is to be a 2-tuple
    Returns list of matplotlib-artist objects"""
    if axis is None:
        axis = plt.gca()

    def f_neg(t, x):
        return -f(t, x)

    artists = []
    tt = np.linspace(*t, steps)
    for ff in (f, f_neg):
        for i in inits:
            # solve_ivp(..., dense_output=True).sol holds a ´OdeSolution´ object
            # which interpolates the solution and allows its evaluation at
            # arbitrary points
            # Returns array with shape(n,) corresponding to the RHS of the
            # given ODE
            sol = solve_ivp(ff, t, i, dense_output=1, **sivp_kwargs).sol
            sol_eval = sol(tt)
            sol_x_ma = np.ma.masked_outside(sol_eval[0], *xbound)
            sol_y_ma = np.ma.masked_outside(sol_eval[1], *ybound)
            artists.append(axis.plot(sol_x_ma, sol_y_ma, **plt_kwargs))
    return artists



def example(which=0):
    """create some example data
    TODO:
        expand
        reorganize if-else as example-dict
    """
    if which == 0:
        f_sym = (x**3 - 3*x*y**2 - 1, 3*x**2*y - y**3)
        f_lam = lambdify((x, y), f_sym)
        return f_sym, f_lam
    elif which == 1:
        def fs(s, n):
            def f(t, x):
                return np.array([1, s*x[1]**n*(1-x[1]**2)])
            return f
        return [fs(s, n) for s in (-1, 1) for n in (1, 2)]
    elif which == 'inits_large':
        return np.array([(i, j) for i in np.arange(-10, 11, 2)
                         for j in np.arange(-1.5, 2, 1)])
    elif which == 'inits_small':
        return np.array([(i, j) for i in np.arange(-2, 3, 1)
                         for j in (1.2, .5, -.5, -1.2)])
