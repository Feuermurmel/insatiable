from tests.ast.utils import check_expression


def test_repr():
    check_expression('type(True)', 'bool')
    check_expression('type(())', 'tuple')
    check_expression('type("foo")', 'str')
    check_expression('type(print)', 'callable')


def test_equality():
    check_expression('type(True) == type(False)', 'True')
    check_expression('type(()) == type((False, True))', 'True')
    check_expression('type("foo") == str', 'True')
    check_expression('type(print) == type("bar")', 'False')
