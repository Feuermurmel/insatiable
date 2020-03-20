import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import List, Set, Any, TextIO, Optional


class CNFExprVar:
    """
    Represents a variable in CNF expression. The variable is encoded the same
    as in a DIMACS CNF file. IDs are positive integers and are negated to
    express that the inverted value of the variable is used in a CNF clause.
    """

    def __init__(self, id):
        self._id = id

    def __invert__(self):
        """
        Invert the variable. This is used to express that the inverted value
        of a variable should be used when the variable is used in a CNF clause.
        """

        return type(self)(-self._id)

    def __repr__(self):
        return f'CNFExprVar({self._id})'


class CNFExpr:
    """
    Represents a complete CNF expression. It contains a list of clauses,
    each clause is a set of references to variables.

    See: https://people.sc.fsu.edu/~jburkardt/data/cnf/cnf.html
    """

    def __init__(self, max_var_id, clauses: List[Set[Any]]):
        self.max_var_id = max_var_id
        self.clauses = clauses


class CNFExprBuilder:
    """
    Helper used to build CNF expressions incrementally.
    """

    def __init__(self):
        self._max_var_id = 0
        self._clauses = []

    def add_variable(self):
        """
        Create and return a new variable, choosing an unused ID automatically.
        """

        self._max_var_id += 1

        return CNFExprVar(self._max_var_id)

    def add_clause(self, *vars):
        """
        Add a clause referencing the specified variables.
        """

        self._clauses.append(set(vars))

    def build(self):
        """
        Create a CNF expression from the clauses added to this builder.
        """

        return CNFExpr(self._max_var_id, self._clauses)


class CNFSolution:
    """
    Represents a solution obtained from running a SAT solver on a CNF
    expression. The solution is represented as a mapping from all CNF
    variables used in the original CNF expression to the boolean values
    assigned to them as part of the solution. The values can be accessed
    using the subscript operator.
    """

    def __init__(self, values_by_id):
        self._values_by_id = values_by_id

    def __getitem__(self, item: CNFExprVar):
        """
        Return the boolean value assigned to the specified CNF variable as
        part of this solution.
        """

        # Invert the value when the id is negative.
        return self._values_by_id[abs(item._id)] != (item._id < 0)


def write_cnf_expr(expr: CNFExpr, file: TextIO):
    """
    Write the specified CNF expression in the DIMACS CNF file format to the
    specified text file.
    """

    print(f'p cnf {expr.max_var_id} {len(expr.clauses)}', file=file)

    for i in expr.clauses:
        parts = sorted((j._id for j in i), key=abs) + [0]

        print(' '.join(map(str, parts)), file=file)


def read_cnf_solution(file, original_expr: CNFExpr) -> Optional[CNFSolution]:
    """
    Read an output file produced by a SAT solver. If the file states a
    solution, it is returned as an `CNFSolution` instance. If the file states
    that there are no solutions, None is returned.

    See: https://dwheeler.com/essays/minisat-user-guide.html
    """

    line = next(file).strip()

    if line == 'SAT':
        values_by_id = {}

        for i in next(file).split():
            id = int(i)

            if not id:
                break

            values_by_id[abs(id)] = id > 0

        if max(values_by_id) != original_expr.max_var_id:
            raise Exception(
                f'Number of variables does not match original expression: '
                f'{max(values_by_id)} != {original_expr.max_var_id}')

        for i in range(1, original_expr.max_var_id + 1):
            if i not in values_by_id:
                raise Exception(f'Missing value for variable: {i}')

        return CNFSolution(values_by_id)
    elif line == 'UNSAT':
        return None
    else:
        raise Exception(f'Invalid first line: {line}')


def run_minisat(expr: CNFExpr, *, print_input_file=False, print_solution_file=False):
    """
    Run MiniSAT, passing it the specified CNF expression as SAT problem to
    solve. Returns the solution found by MiniSAT, if any, as a `CNFSolution`
    instance. If no solution is found, None is returned.

    The command `minisat` needs to be on `$PATH`.

    :param expr: The CNF expression to solve.
    :param print_input_file:
        Whether to print the DIMACS CNF input file to stderr before running
        MiniSAT.
    :param print_solution_file:
        Whether to print the output file to stderr before reading it.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = pathlib.Path(temp_dir)
        in_path = temp_dir / 'in'
        out_path = temp_dir / 'out'

        with in_path.open('w', encoding='utf-8') as file:
            write_cnf_expr(expr, file)

        if print_input_file:
            with in_path.open(encoding='utf-8') as file:
                shutil.copyfileobj(file, sys.stderr)

        exit_code = subprocess.call(['minisat', str(in_path), str(out_path)])

        assert exit_code in [10, 20]

        if print_solution_file:
            with out_path.open(encoding='utf-8') as file:
                shutil.copyfileobj(file, sys.stderr)

        with out_path.open(encoding='utf-8') as file:
            return read_cnf_solution(file, expr)
