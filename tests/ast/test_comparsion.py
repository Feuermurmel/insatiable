from tests.ast.utils import check_expression


def test_equals():
    check_expression('True == True', 'True')
    check_expression('True == False', 'False')

    check_expression("'foo' == 'foo'", 'True')
    check_expression("'foo' == 'bar'", 'False')

    check_expression("((), ('a', 'b')) == ((), ('a', 'b'))", 'True')
    check_expression("((), ('a', 'b', ())) == ((), ('a', 'b'))", 'False')

    check_expression('False == ()', 'False')
    check_expression("'foo' == ('foo',)", 'False')


def test_not_equals():
    check_expression('True != False', 'True')
    check_expression('False != ()', 'True')
    check_expression('() != ()', 'False')


def test_multiple_ops():
    check_expression("'a' == 'a' != 'b'", 'True')
    check_expression("'a' == 'b' == 'b'", 'False')
    check_expression("'a' != 'b' != 'a'", 'True')


def test_operands_only_evaluated_once(runner):
    runner.run(
        """
        def get(value, message):
            print(message)
            return value

        get('a', '1') == get('a', '2') == get('a', '3')
        get('a', '4') == get('b', '5') == get('a', '6')
        """)

    assert runner.output.count('1') == 1
    assert runner.output.count('2') == 1
    assert runner.output.count('3') == 1
    assert runner.output.count('4') == 1
    assert runner.output.count('5') == 1
    assert runner.output.count('6') == 0
