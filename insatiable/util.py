import sys


def log(message):
    print(message, file=sys.stderr, flush=True)


class UserError(Exception):
    def __init__(self, message, *args):
        super().__init__(message.format(*args))


class Hashable:
    def __eq__(self, other):
        return type(self) is type(other) \
               and self._hashable_key() == other._hashable_key()

    def __hash__(self):
        return hash(self._hashable_key())

    def _hashable_key(self):
        raise NotImplementedError()
