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
