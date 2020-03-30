def test_pack(runner):
    runner.run(
        """
        (a, b) = ('a', 'b')
        print((a, b))
        """)

    runner.check_output_line("('a', 'b')")


def test_pack_starred(runner):
    runner.run(
        """
        (a, *b) = ('a', 'b')
        print((a, b))
        """)

    runner.check_output_line("('a', ('b',))")

    runner.run(
        """
        (a, *b, c) = ('a', 'b')
        print((a, b, c))
        """)

    runner.check_output_line("('a', (), 'b')")


def test_error_pack_too_few(runner):
    runner.run(
        """
        (a, b, c) = ('a', 'b')
        """)

    assert 'with exactly 3 elements' in runner.output


def test_error_pack_too_many(runner):
    runner.run(
        """
        (a, b, c) = ('a', 'b', 'c', 'd')
        """)

    assert 'with exactly 3 elements' in runner.output


def test_error_pack_starred_too_few(runner):
    runner.run(
        """
        (a, *b, c) = ('a',)
        """)

    assert 'with at least 2 elements' in runner.output


def test_error_pack_not_a_tuple(runner):
    runner.run(
        """
        (a, *b, c) = 'a'
        """)

    assert 'only unpack a tuple' in runner.output
