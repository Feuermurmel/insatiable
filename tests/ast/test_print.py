def test_print(runner):
    runner.run('print("foo")')
    assert runner.output == 'foo\n'

    runner.run('print("foo", "bar")')
    assert runner.output == 'foo bar\n'

    runner.run('print(True, ())')
    assert runner.output == 'True ()\n'

    runner.run('print((True, ()))')
    assert runner.output == '(True, ())\n'
