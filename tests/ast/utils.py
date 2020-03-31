import io
import textwrap

from insatiable.ast import insat_module_from_string, solve_module


"""
Piece of code which is prepended to source strings executed using a `Runner`
 instance.
"""
_default_preamble = textwrap.dedent(
    """
    from insatiable import invariant
    """)


class Runner:
    def __init__(self):
        self._expect_unsat = False
        self.output = None

    def expect_unsat(self):
        self._expect_unsat = True

    def run(self, source):
        source = _default_preamble + textwrap.dedent(source)
        module = insat_module_from_string(source, '<string>')
        solution = solve_module(module)

        if solution is None:
            if not self._expect_unsat:
                raise Exception('Expected the program to be satisfiable.')
        else:
            if self._expect_unsat:
                raise Exception('Expected the program to be unsatisfiable.')

            buffer = io.StringIO()

            solution.run(buffer)

            self.output = buffer.getvalue()

        self._expect_unsat = False

    def check_output_line(self, line: str):
        assert line in self.output.splitlines()


def check_expression(expr: str, expected_output: str):
    """
    Return a string which results from passing the specified expression to
    `print()` and capturing the output.
    """

    runner = Runner()

    runner.run(f'print({expr})')

    assert runner.output.strip() == expected_output
