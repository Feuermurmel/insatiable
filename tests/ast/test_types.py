from tests.ast.utils import check_expression


def test_repr():
    check_expression('type(True)', 'bool')
    check_expression('type(())', 'tuple')
    check_expression('type("foo")', 'str')
    check_expression('type(print)', 'callable')
