import pytest


def test_error_raised(runner):
    # Check correct behavior for error found during parsing.
    with pytest.raises(SyntaxError):
        runner.run(
            """
            print()
            print(a,,)
            """)

    # Check correct behavior for error found during later analysis.
    with pytest.raises(SyntaxError):
        runner.run(
            """
            *a, *b = ()
            """)
