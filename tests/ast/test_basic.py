def test_empty_module(runner):
    runner.run('')

    assert runner.output == ''


def test_invariant(runner):
    runner.run(
        """
        a = bool()
        print(a)
        invariant(a)
        """)

    runner.check_output_line('True')

    runner.run(
        """
        a = bool()
        print(a)
        invariant(not a)
        """)

    runner.check_output_line('False')


def test_unsatisfiable(runner):
    runner.expect_unsat()

    runner.run(
        """
        invariant(False)
        """)

    runner.expect_unsat()

    runner.run(
        """
        a = bool()
        invariant(a and not a)
        """)
