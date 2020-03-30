def test_values(runner):
    """
    Check that values from different slices assigned to the same variable are
    correctly kept apart.
    """

    runner.run(
        """
        a = bool()

        if a:
            b = 'foo'
        else:
            b = 'bar'

        print(b)
        invariant(a)
        """)

    runner.check_output_line('foo')


def test_values_functions(runner):
    runner.run(
        """
        def x():
            return 'x'

        def y():
            return 'y'

        a = bool()

        if a:
            b = x
        else:
            b = y

        print(b())
        invariant(a)
        """)

    runner.check_output_line('x')


def test_return(runner):
    """
    Check that return values are kept apart.
    """

    runner.run(
        """
        b = bool()

        def f():
            if b:
                return 'foo'
            else:
                return 'bar'

        print(f())
        invariant(b)
        """)

    runner.check_output_line('foo')
