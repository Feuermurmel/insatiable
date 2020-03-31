from tests.ast.utils import check_expression


def test_print():
    check_expression("'abc'", 'abc')


def test_print_tuples():
    # In the end, Python's __str__() is called, just check that we get what
    # we expected.
    check_expression('("a", "b")', "('a', 'b')")
