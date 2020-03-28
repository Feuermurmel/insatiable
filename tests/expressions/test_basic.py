from insatiable.expressions import false, true, var, _nand


def n(*args):
    return _nand(*args)


a = var.a
b = var.b
c = var.c


def test_literals():
    assert false == n()
    assert true == n(n())


def test_operations():
    assert ~a == n(a)
    assert a & b == n(n(a, b))
    assert a | b == n(n(a), n(b))
    assert a / b == n(n(a, n(b)))


def test_simplify():
    assert ~~a == a
    assert ~(a & b) == n(a, b)
    assert ~(a & b & c) == n(a, b, c)

    assert a & true == a
    assert a & false == false
    assert a | true == true
    assert a | false == a

    assert a & a == a
    assert a & ~a == false
    assert a | a == a
    assert a | ~a == true


def test_simplify_more():
    assert (a & b) & (a & c) == a & b & c
    assert (a & b) & (~a & c) == false
    assert (a | b) | (a | c) == a | b | c
    assert (a | b) | (~a | c) == true

    assert (a & b) & ~(a & b) == false
    assert (a | b) & ~(a | b) == false
    assert (a & b) | ~(a & b) == true
    assert (a | b) | ~(a | b) == true
