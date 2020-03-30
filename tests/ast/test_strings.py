
def test_print(runner):
    runner.run('print("abc")')
    assert runner.output.strip() == 'abc'
