
def test_print(runner):
    runner.run('print("abc")')
    assert runner.output.strip() == 'abc'


def test_print_tuples(runner):
    # In the end, Python's __str__() is called, just check that we get what
    # we expected.
    runner.run('print(("a", "b"))')
    assert runner.output.strip() == "('a', 'b')"
