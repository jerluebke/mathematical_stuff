#coding=utf-8
"""
Needs to be rewritten!


Constants:
    SPERICAL_COORDS
    ZYLINDIRCAL_COORDS


Classes:

Tensor:
    Abstraction of SymTensor2D
    To be implemented later


SymTensor2D:
    Inherited from sympy.ImmutableMatrix
    Represents 2nd order tensor as symmetric 3x3 matrix
    e.g. moment of inertia, quadrupole moment

    Properties:
        elements
        element_definition
        property_distribution
        coordinate_system

    Functions:
        make_elements
        show_as_matrix
        show_elementwise


TensorElement:
    Properties:
        position
        input_data
        symbolic_calculated_data
        explicit_solved_data

    Functions:
        eval_input
        calculate_symbolic
        calculate_explicit


Integral3D:
    Inherited from sympy.integral
    Calculates or gets saved jacobian determinant and performs integration over
    the whole space

    Properties:
        coordinate_system
        jacobian_determinant
        general_form
        general_solution
        explicit_solution


Jacobian:
    Inherited from sympy.ImmutableMatrix
    Functions:
        det


Functions:
    kronecker_delta
    coord_transform


Requirements:
    sympy
"""

from sympy import Integral, ImmutableMatrix, Symbol
from sympy import Heaviside
from sympy import diff, symbols, sympify, simplify
from sympy import sin, cos
from sympy import pi
from sympy import init_printing

init_printing(use_unicode=True)


#############
# CONSTANTS #
#############

# Symbols
x, y, z = symbols('x, y, z')
r, rho, phi, theta = symbols('r, rho, phi, theta')

# Mapping for coordinate substitution
ZYL_MAP = (rho, phi, z)
SPH_MAP = (r, theta, phi)

# Transformations
CARTESIAN = ((x,
              y,
              z),)

ZYLINDRICAL = ((rho*cos(phi),
                rho*sin(phi),
                z),
               ZYL_MAP)

SPHERICAL = ((r*sin(theta)*cos(phi),
              r*sin(theta)*sin(phi),
              r*cos(theta)),
             SPH_MAP)

TRANSFORM = {'CAR': CARTESIAN,
             'ZYL': ZYLINDRICAL,
             'SPH': SPHERICAL}

# Jacobian Determinants for common coordinate systems
JD = {'CAR': 1,
      'SPH': r**2*sin(theta),
      'ZYL': rho}



###########
# Classes #
###########

class SymTensor2D(ImmutableMatrix):
    pass



class TensorElement:
    pass



class Integral3D(Integral):
    """Represents an unevaluated 3d integral

    Properties:
        function - inherited from integral
        antiderivative
        solution - call doit
    """
    def __new__(cls, func, coords, sym_x, sym_y, sym_z, *args, **kwargs):
        """Tidy this doc_string up!!!

        if you use one of the predefined coord systems, make sure to follow
        this convention:
            CART = {x, y, z}
            ZYL = {rho, phi, z}
            SPH = {r, theta, phi}
        where rho is the distance from the z axis, theta is the polar angel and
        phi is the azimuthal angel.

        otherwise specify your coordinates in a tuple like this:
            coords = ((x(q_1, _2, _3),
                       y(q_1, _2, _3),
                       z(q_1, _2, _3),
                      (q_1, q_2, q_3))  # <- defines order

        Don\'t forget to set
            transform = True
        when your input is not already
        expressed in the desired coordinate system!

        In case of transform = True func.free_symbols will be converted to list
        and sorted alphabetically. Then these symbols are mapped one by one to
        coords tuple

        With transform = True is NOT YET implemented to adjust the integration
        limits accordingly, so do it yourself!
        """
        if not coords:
            coords = 'CART'     # set default to cartesian - this doesn't
                                # change the input function
        if coords in ('SPHERICAL', 'SPH', 'ZYLINDIRCAL', 'ZYL', 'CARTESIAN', 'CART'):
            jacobian_determinant = JD[coords[:3]]
            coords = TRANSFORM[coords[:3]]
        else:                                                   # custom coordinates
            jacobian_determinant = Jacobian(coords[0]).det()    # propably some
                                                                # saftey against
                                                                # bad input would
                                                                # be nice ...
        sym_list = sympify((sym_x, sym_y, sym_z))

        if 'transform' in kwargs:
            if kwargs['transform'] is True and coords != CARTESIAN:
                func_sym = list(func.free_symbols)          # sort alphabetically
                func_sym.sort(key=lambda s: str(s)[0])      # for 1-1 substitution
                func = coord_transform(func, func_sym, coords[0])   # <- in there

                # substitute integration varibles
                def sub_var(var_iter, new_var):
                    try:
                        var_iter = list(var_iter)
                        var_iter[0] = new_var
                    except TypeError:
                        var_iter = new_var
                    return var_iter

                sym_list = [sub_var(o, n) for o, n in zip(sym_list, coords[1])]

            del kwargs['transform']     # needs to be removed because kwargs is
                                        # later passed to super().__new__,
                                        # which doesn't understand this keyword
        func = func * jacobian_determinant
        return super().__new__(cls, func, *sym_list, **kwargs)

    @property
    def antiderivative(self):
        return self.func(self.function, None, self.function.free_symbols).doit()



class Jacobian(ImmutableMatrix):
    """Consider some function f:R^n -> R^m
    Then the Jacobian is
        J = (df_i/dx_j)_ij in M(m x n, R)
    That is the Matrix of all partial derivations of f

    For instanciation call Jacobian(f), where f should a tuple (f_1, ..., f_m)
    If the input is no sympy expression it is converted to such
    """
    def __new__(cls, f, *args, **kwargs):
        """returns instance of sympy.ImmutableMatrix"""
        if isinstance(f, str):
            f = sympify(f)  # input of type str may need to be sympified two
                            # times
                            # type(sympify('x', 'y')) == tuple
                            # type(sympify(sympify('x', 'y'))) == sympy...Tuple
        f = sympify(f)
        J = [[diff(f_i, x_j) for x_j in f.free_symbols] for f_i in f]
        return ImmutableMatrix.__new__(cls, J, *args, **kwargs)

    def det(self, **kwargs):
        """returns Determinant of Matrix (simplified) (sympy expression)"""
        return super().det().simplify()



class H(Heaviside):
    """Modification of Heaviside function to adjust limits of integrals by
    multiplying"""
    def __new__(cls, arg, **options):
        return super().__new__(cls, arg, H0=1, **options)

    # The following are propably needed
    def __mul__(self, mul):
        """self * mul"""
    def __rmul__(self, mul):
        """mul * self"""
    def __imul__(self, mul):
        """self *= mul"""



####################
# Module Functions #
####################

def kronecker_delta(i, j):
    """pretty self explaining
    this is a simplified solution for this module
    for a more advanced implementation see sympy.KroneckerDelta

    returns Integer
    """
    return 1 if i == j else 0

kd = kronecker_delta


def coord_transform(func, symbols, transform):
    """Apply coordinate transformation on given function by iterativly
    substituting symbols[i] with transform[i]

    Note: symbols and transform need to be in fitting order

    returns sympified function
    """
    # TODO
    # transform integral limits 
    try:
        if not func.free_symbols:
            raise ValueError('no free symbols')
    except AttributeError:
        func = sympify(func)

    for i, s in enumerate(symbols):
        if not s in func.free_symbols:
            raise ValueError('symbols doesn\'t match with func.free_symbols')
        func = func.subs(s, transform[i])
    return func
