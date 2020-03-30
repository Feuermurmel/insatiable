import subprocess
import textwrap


def test_successful(tmp_path):
    test_file_path = tmp_path / 'test.insat'

    test_file_path.write_text(
        textwrap.dedent(
            """
            print('Hello')
            """))

    result = subprocess.check_output(['insatiable', test_file_path])

    # Nothing than the output of the program should appear on stdout.
    # Everything else needs to go to stderr.
    assert result == b'Hello\n'


def test_syntax_error(tmp_path):
    test_file_path = tmp_path / 'test.insat'

    test_file_path.write_text(
        textwrap.dedent(
            """
            print(,,)
            """))

    result = subprocess.run(
        ['insatiable', test_file_path],
        stderr=subprocess.PIPE)

    # The command should fail and produce an error message.
    assert result.returncode != 0
    assert b'SyntaxError' in result.stderr


def test_unsatisfiable(tmp_path):
    test_file_path = tmp_path / 'test.insat'

    test_file_path.write_text(
        textwrap.dedent(
            """
            from insatiable import invariant
            invariant(False)
            """))

    result = subprocess.run(
        ['insatiable', test_file_path],
        stderr=subprocess.PIPE)

    # The command should fail and produce an error message.
    assert result.returncode != 0
    assert b'No solutions found' in result.stderr
