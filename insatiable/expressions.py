from typing import List, Mapping, Optional

from insatiable.cnf import run_minisat, CNFExprBuilder, CNFExprVar, CNFExpr, \
    CNFSolution
from insatiable.util import Hashable


class Expr(Hashable):
    """
    Represents a boolean expression in terms of a number of variables.
    """

    def __and__(self, other):
        """
        Return the conjunction `self AND other` of the two expressions.
        """

        return ~(~self | ~other)

    def __or__(self, other):
        """
        Return the disjunction `self OR other` of the two expressions.
        """

        return _nand(~self, ~other)

    def __xor__(self, other):
        """
        Return the exclusive or `self XOR other` of the two expressions.
        """

        return (self | other) & ~(self & other)

    def __truediv__(self, other):
        """
        Return the difference `self WITHOUT other` of the two expressions.
        """

        return self & ~other

    def __invert__(self):
        """
        Return the negation `NOT self` of the two expressions.
        """

        return _nand(self)


class Var(Expr):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'vars.{self.name}'

    def _hashable_key(self):
        return self.name


class Nand(Expr):
    def __init__(self, children: List[Expr]):
        self.children = children

    def __repr__(self):
        if self == true:
            return 'true'
        elif self == false:
            return 'false'
        else:
            return 'nand({})'.format(', '.join(map(repr, self.children)))

    def _hashable_key(self):
        return frozenset(self.children)


def _expand_nand_child(child):
    # By accident, this will also expand true to an empty list.
    if isinstance(child, Nand) and len(child.children) == 1:
        child_child, = child.children

        if isinstance(child_child, Nand):
            return child_child.children

    return [child]


def _simplify_nand_children(children):
    new_children = []

    for i in children:
        # Ignore other children if one of them is constant false.
        if i == false:
            return [i]

        # Ignore children which are equivalent to other children.
        if i not in new_children:
            new_children.extend(_expand_nand_child(i))

    return new_children


def _nand(*children):
    # Remove redundant children.
    children = _simplify_nand_children(children)

    # Simplify Nand(Nand(x)) to x.
    if len(children) == 1:
        child, = children

        if isinstance(child, Nand) and len(child.children) == 1:
            child_child, = child.children

            return child_child

    return Nand(children)


false = _nand()
true = ~false


class Vars:
    """
    Helper used to implement the member `var` of this module.
    """

    def __call__(self, name):
        return Var(name)

    def __getattr__(self, item):
        return self(item)


"""
Each attribute accessed on this instance returns a `Var` instance with the 
name of the accessed attribute. Can also be called with the variable name
specified as string with the same result.
"""
var = Vars()


class CompiledExpr:
    """
    Used to combine a CNF expression which has been produced from an `Expr`
    instance with a symbol table mapping from variables of the original
    expression to CNF variables of the converted expression.
    """

    def __init__(self, cnf_expr: CNFExpr, cnf_vars_by_var: Mapping[Var, CNFExprVar]):
        self.cnf_expr = cnf_expr
        self.cnf_vars_by_var = cnf_vars_by_var

    def resolve_solution(self, solution: CNFSolution) -> Mapping[Var, bool]:
        """
        Use the symbol table in this instance to resolve the specified
        solution to a map from variables of the original expression to
        booleans.
        """

        return {k: solution[v] for k, v in self.cnf_vars_by_var.items()}


def to_cnf(e: Expr) -> CompiledExpr:
    """
    Convert the specified `Expr` instance to a CNF expression.

    The returned expression possibly contains more variables than the
    original expression. This is the result of applying the Tseytin
    transformation to the expression to prevent an exponential increase of
    clauses when converting the sub-expressions to CNF clauses.

    See: https://en.wikipedia.org/wiki/Tseytin_transformation
    """

    builder = CNFExprBuilder()
    cnf_vars_by_expr = {}

    def walk(e) -> CNFExprVar:
        # Special case for simple inversions, which do not need an additional
        # variable.
        if isinstance(e, Nand) and len(e.children) == 1:
            child, = e.children

            return ~walk(child)

        cnf_var = cnf_vars_by_expr.get(e)

        if cnf_var is None:
            cnf_var = builder.add_variable()
            cnf_vars_by_expr[e] = cnf_var

            if isinstance(e, Var):
                # Nothing more than allocating the variable is needed.
                pass
            elif isinstance(e, Nand):
                # We can assume that the children of a Nand instance do
                # not contain redundant terms. The _nand() constructor
                # removes them.
                cnf_children = [walk(i) for i in e.children]

                # Add clauses needed for this nand expression.
                builder.add_clause(~cnf_var, *(~i for i in cnf_children))

                for i in cnf_children:
                    builder.add_clause(cnf_var, i)
            else:
                assert False

        return cnf_var

    # Require the root variable to be true.
    builder.add_clause(walk(e))

    cnf_vars_by_var = \
        {k: v for k, v in cnf_vars_by_expr.items() if isinstance(k, Var)}

    return CompiledExpr(builder.build(), cnf_vars_by_var)


def solve_expr(expr: Expr) -> Optional[Mapping[Var, bool]]:
    """
    Solve an `Expr` instance and return a map from the variables used in the
    expression to booleans. If no solution is found, None is returned.
    """

    compiled_expr = to_cnf(expr)

    cnf_solution = run_minisat(
        compiled_expr.cnf_expr,
        print_input_file=True,
        print_solution_file=True)

    if cnf_solution is None:
        return None

    return compiled_expr.resolve_solution(cnf_solution)
