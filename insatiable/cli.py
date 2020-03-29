import argparse
import pathlib
import sys

from insatiable.ast import load_insat_module, run_module
from insatiable.util import log, UserError


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'file',
        type=pathlib.Path)

    return parser.parse_args()


def main(file):
    run_module(load_insat_module(file))


def entry_point():
    try:
        main(**vars(parse_args()))
    except KeyboardInterrupt:
        log('Operation interrupted.')
        sys.exit(1)
    except UserError as e:
        log(f'error: {e}')
        sys.exit(2)
