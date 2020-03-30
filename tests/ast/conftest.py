import pytest

from tests.ast.utils import Runner


@pytest.fixture
def runner():
    return Runner()
