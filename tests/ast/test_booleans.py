def test_booleans(runner):
    runner.run('print(True)')
    assert runner.output.strip() == 'True'

    runner.run('print(False)')
    assert runner.output.strip() == 'False'


def test_bool(runner):
    runner.run(
        """
        a = bool()
        print(a or not a)
        """)

    runner.check_output_line('True')
