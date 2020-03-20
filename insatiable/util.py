class Hashable:
    def __eq__(self, other):
        return type(self) is type(other) \
               and self._hashable_key() == other._hashable_key()

    def __hash__(self):
        return hash(self._hashable_key())

    def _hashable_key(self):
        raise NotImplementedError()
