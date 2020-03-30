import pytest


def test_create(runner):
    runner.run(
        """
        a = 'a', 'b'
        print(a)
        """)

    runner.check_output_line("('a', 'b')")


def test_error_pack_multiple_starred(runner):
    with pytest.raises(SyntaxError):
        runner.run(
            """
            (*b, *c) = ('a', 'b')
            """)


def test_tuple_unpack(runner):
    runner.run(
        """
        a = 'a', 'b'
        b = ((), *a, *(), *a)
        print(b)
        """)

    runner.check_output_line("((), 'a', 'b', 'a', 'b')")


def test_unpack_non_tuple(runner):
    runner.run(
        """
        b = ((), *False)
        """)

    assert 'Can only unpack a tuple' in runner.output
