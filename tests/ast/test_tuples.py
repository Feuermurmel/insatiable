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
