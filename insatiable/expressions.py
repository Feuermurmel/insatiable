import sys
import weakref
from typing import Mapping, Optional, MutableMapping, FrozenSet

from insatiable.cnf import run_minisat, CNFExprBuilder, CNFExprVar, CNFExpr, \
    CNFSolution


class Expr:
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


class Nand(Expr):
    def __init__(self, children: FrozenSet[Expr]):
        self.children = children

    def __repr__(self):
        if self == true:
            return 'true'
        elif self == false:
            return 'false'
        else:
            children_str = ' & '.join(map(repr, self.children))

            return f'~({children_str})'


_var_instances: MutableMapping[str, Var] = \
    weakref.WeakValueDictionary()

_nand_instances: MutableMapping[FrozenSet[Expr], Nand] = \
    weakref.WeakValueDictionary()


def _single_nand_child(expr: Expr):
    if isinstance(expr, Nand) and len(expr.children) == 1:
        i_child, = expr.children

        return i_child
    else:
        return None


def _simplify_nand_children(children):
    # Ignore children which are equivalent to other children.
    new_children = set()

    for i in children:
        i_child = _single_nand_child(i)

        # By accident, this will also expand true to an empty list.
        if isinstance(i_child, Nand):
            new_children |= i_child.children
        else:
            new_children.add(i)

    for i in new_children:
        if isinstance(i, Nand):
            # Check if we have a term which will be false in all cases where
            # even a subset of the other terms are true.
            if i.children <= new_children:
                return [false]

    return new_children


def _nand(*children):
    # Remove redundant children.
    children = frozenset(_simplify_nand_children(children))

    # Simplify Nand(Nand(x)) to x.
    if len(children) == 1:
        child, = children
        child_child = _single_nand_child(child)

        if child_child is not None:
            return child_child

    instance = _nand_instances.get(children)

    if instance is None:
        instance = _nand_instances[children] = Nand(children)

    return instance


def and_all(exprs) -> Expr:
    return ~_nand(*exprs)


def or_all(exprs) -> Expr:
    return _nand(*(~i for i in exprs))


def ite(condition, then, or_else):
    return then & condition | or_else / condition


false = _nand()
true = ~false


class Vars:
    """
    Helper used to implement the member `var` of this module.
    """

    def __call__(self, name):
        instance = _var_instances.get(name)

        if instance is None:
            instance = _var_instances[name] = Var(name)

        return instance

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

    def resolve_solution(self, solution: CNFSolution) -> 'ExprSolution':
        """
        Use the symbol table in this instance to resolve the specified
        solution to a map from variables of the original expression to
        booleans.
        """

        return ExprSolution(
            {k: solution[v] for k, v in self.cnf_vars_by_var.items()})


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
        if (child := _single_nand_child(e)) is not None:
            return ~walk(child)

        if (cnf_var := cnf_vars_by_expr.get(e)) is None:
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


class ExprSolution:
    """
    Represents a solution which satisfies an expression. The solution assigns
    a boolean value to each variable used in the original expression.
    """

    def __init__(self, values_by_var: Mapping[Var, bool]):
        self.values_by_var = values_by_var

    def __call__(self, expr: Expr) -> bool:
        """
        Evaluate an expression by replacing each variable in the specified
        expression with its value from the solution.

        The solution only contains a definitive value for variables which
        appeared in the expression which was solved. All other variables are
        assumed to have the value `False`.
        """

        if isinstance(expr, Var):
            return self.values_by_var.get(expr, False)
        elif isinstance(expr, Nand):
            return not all(self(i) for i in expr.children)
        else:
            assert False


def solve_expr(expr: Expr) -> Optional[ExprSolution]:
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


def dump_compiled_expr(expr: CompiledExpr, file=sys.stdout):
    """
    Write all clauses of the CNF expression contained in the specified
    compiled expression to the specified file. CNF variables that were
    already present in the original boolean expression will be mapped to
    their original name.
    """

    vars_names_cnf_var_id = {
        cnf_var.id: var.name
        for var, cnf_var in expr.cnf_vars_by_var.items()}

    def map_var(v: CNFExprVar):
        prefix = '~' if v.id < 0 else ''
        name_or_id = vars_names_cnf_var_id.get(abs(v.id), abs(v.id))

        # Sort variables by their mapped name or ID, sorting variables with a
        # name before the others.
        sort_key = isinstance(name_or_id, int), name_or_id

        # The tuples will be sorted by their first element. The second element
        # is used in the output.
        return sort_key, f'{prefix}{name_or_id}'

    for i in expr.cnf_expr.clauses:
        print(' '.join(j for _, j in sorted(map(map_var, i))), file=file)
