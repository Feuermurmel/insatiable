import functools
import itertools
from typing import Mapping, Optional

from insatiable.expressions import Expr, var, false, true, Var, solve_expr
from insatiable.util import Hashable


class IntExpr(Hashable):
    def __init__(self, *, size):
        self.size = size

    def __add__(self, other):
        return IntAddition(self, other)


class IntVar(IntExpr):
    def __init__(self, name, size):
        super().__init__(size=size)

        self.name = name

    def __repr__(self):
        return f'var({self.name}, {self.size})'

    def _hashable_key(self):
        return self.name, self.size


class IntConstant(IntExpr):
    def __init__(self, value):
        assert value >= 0

        size = len(format(100, 'b'))

        super().__init__(size=size)

        self.value = value

    def __repr__(self):
        return f'const({self.value})'

    def _hashable_key(self):
        return self.value


class IntAddition(IntExpr):
    def __init__(self, left, right):
        super().__init__(size=max(left.size, right.size) + 1)

        self.left = left
        self.right = right

    def __repr__(self):
        return f'({self.left} + {self.right})'

    def _hashable_key(self):
        return self.left, self.right


class IntEquals(IntExpr):
    def __init__(self, left, right):
        super().__init__(size=1)

        self.left = left
        self.right = right

    def __repr__(self):
        return f'({self.left} == {self.right})'

    def _hashable_key(self):
        return self.left, self.right


def int_var(name, size):
    return IntVar(name, size)


def int_const(value: int):
    return IntConstant(value)


def int_equals(left, right):
    return IntEquals(left, right)


class CompiledIntExpr:
    def __init__(self, expr: Expr, expr_vars_by_int_var):
        self.expr = expr
        self.expr_vars_by_int_var = expr_vars_by_int_var

    def resolve_solution(self, solution: Mapping[Var, bool]) -> Mapping[IntVar, int]:
        """
        Use the symbol table in this instance to resolve the specified
        solution to a map from variables of the original expression to
        booleans.
        """

        def get_value(expr_vars):
            # Convert the booleans to the string of a binary number and parse
            # the number.
            return int(''.join('01'[solution[i]] for i in expr_vars), 2)

        return {k: get_value(v) for k, v in self.expr_vars_by_int_var.items()}


def _half_add(a, b):
    return a & b, a ^ b


def _full_add(a, b, c):
    carry1, half_sum = _half_add(a, b)
    carry2, sum = _half_add(half_sum, c)

    return carry1 | carry2, sum


def to_expr(expr: IntExpr) -> CompiledIntExpr:
    """
    Create a boolean expression from this int expression. The returned
    expression will be true whenever the int expression is non-zero.
    """

    exprs_by_int_expr = {}

    def walk(e):
        if isinstance(e, IntVar):
            exprs = exprs_by_int_expr.get(e)

            # If necessary, create new boolean variables for the bits of the
            # int variable.
            if exprs is None:
                exprs = [var(f'{e.name}.{i}') for i in range(e.size)]
                exprs_by_int_expr[e] = exprs

            return exprs
        elif isinstance(e, IntConstant):
            return [{'0': false, '1': true}[i] for i in format(e.value, 'b')]
        elif isinstance(e, IntAddition):
            left_exprs = walk(e.left)
            right_exprs = walk(e.right)

            columns = itertools.zip_longest(
                reversed(left_exprs),
                reversed(right_exprs),
                fillvalue=false)

            def iter_sum_exprs():
                carry = false

                for a, b in columns:
                    carry, sum = _full_add(carry, a, b)

                    yield sum

                yield carry

            return list(reversed(list(iter_sum_exprs())))
        elif isinstance(e, IntEquals):
            left_exprs = walk(e.left)
            right_exprs = walk(e.right)

            columns = itertools.zip_longest(
                reversed(left_exprs),
                reversed(right_exprs),
                fillvalue=false)

            expr = true

            # Compare all bits of the numbers pair-wise.
            for a, b in columns:
                expr &= ~(a ^ b)

            return [expr]
        else:
            assert False

    # The final expression to which the specified int expression was
    # converted. This boolean expression is true whenever the int expression
    # is non-zero.
    root_expr = functools.reduce(lambda x, y: x | y, walk(expr), false)

    expr_vars_by_int_var = \
        {k: v for k, v in exprs_by_int_expr.items() if isinstance(k, IntVar)}

    return CompiledIntExpr(root_expr, expr_vars_by_int_var)


def solve_int_expr(expr: IntExpr) -> Optional[Mapping[IntVar, int]]:
    compiled_int_expr = to_expr(expr)
    expr_solution = solve_expr(compiled_int_expr.expr)

    if expr_solution is None:
        return None

    return compiled_int_expr.resolve_solution(expr_solution)
