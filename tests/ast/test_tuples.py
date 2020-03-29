def test_create(runner):
    runner.run(
        """
        a = True, False
        print(a)
        """)

    runner.check_output_line('(True, False)')
