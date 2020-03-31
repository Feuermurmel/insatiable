from tests.ast.utils import check_expression


def test_booleans():
    check_expression('True', 'True')
    check_expression('False', 'False')


def test_bool(runner):
    runner.run(
        """
        a = bool()
        print(a or not a)
        """)

    runner.check_output_line('True')


def test_truthyness():
    """
    Check that expressions are correctly interpreted as being "truthy" or
    "falsy".
    """

    def check(expr: str, truthyness: bool):
        check_expression(f'not not {expr}', str(truthyness))

    check("True", True)
    check("False", False)

    # Check that only non-empty strings are "truthy".
    check("''", False)
    check("'foo'", True)

    # Check that only non-empty tuples are "truthy".
    check("()", False)
    check("('',)", True)
    check("(True, False)", True)
