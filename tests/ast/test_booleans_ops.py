from tests.ast.utils import check_expression


def test_and(runner):
    runner.run(
        """
        def foo(value, message):
            print(message)
            return value
            
        foo(True, 'left') and foo(True, 'right') 
        """)

    # Both expressions should be evaluated.
    assert 'left' in runner.output
    assert 'right' in runner.output


def test_and_short_circuit(runner):
    runner.run(
        """
        def foo(value, message):
            print(message)
            return value

        foo(False, 'left') and foo(True, 'right') 
        """)

    # Only the left expression should be evaluated.
    assert 'left' in runner.output
    assert 'right' not in runner.output


def test_chain():
    check_expression('True and True and True', 'True')
    check_expression('True and False and True', 'False')


def test_non_booleans():
    check_expression("True and 'a'", 'a')
    check_expression("False and 'a'", 'False')

    check_expression("True or 'a'", 'Tr'
                                    'ue')
    check_expression("False or 'a'", 'a')
