from dhllinalg.bla import Vector


def test_add():
    x = Vector(3)
    y = Vector(3)

    for i in range(len(x)):
        x[i] = i
    y[:] = 2

    for i in range(len(y)):
        assert y[i] == 2
