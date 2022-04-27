import numpy as np
import itertools


def G(row_s: np.array, temp: float):
    # Exercise 1
    asserts([row_s], [temp])
    return np.exp((1 / temp) * row_s[:-1] @ row_s[1:])


def F(row_s: np.array, row_t: np.array, temp: float):
    # Exercise 2
    asserts([row_s, row_t], [temp])
    return np.exp((1 / temp) * (row_s @ row_t))


def ex3():
    n = 2
    perms = list(itertools.product([-1, 1], repeat=n ** 2))  # [(-1,-1,-1,-1),(1,-1,-1,-1),(-1,1,-1,-1)...]
    temps = [1, 1.5, 2]
    ztemps = []
    for temp in temps:
        """
        -----
        x1 x2
        x3 x4
        -----
        x1 -> x2 | x3
        x2 -> x4
        x3 -> x4
        x4 ->
        """
        ztemp = sum([F(np.array([x1, x1, x2, x3]),
                       np.array([x2, x3, x4, x4]), temp)
                     for (x1, x2, x3, x4) in perms])
        ztemps.append(ztemp)
    return dict(zip(temps, ztemps))


def ex4():
    n = 3
    perms = list(itertools.product([-1, 1], repeat=n ** 2))
    temps = [1, 1.5, 2]
    ztemps = []
    for temp in temps:
        """
        --------
        x1 x2 x3
        x4 x5 x6
        x7 x8 x9
        --------
        x1 -> x2 | x4
        x2 -> x3 | x5
        x3 -> x6
        x4 -> x5 | x7
        x5 -> x6 | x8
        x6 -> x9
        x7 -> x8
        x8 -> x9
        X9 -> 
        """
        ztemp = sum([F(np.array([x1, x1, x2, x2, x3, x4, x4, x5, x5, x6, x7, x8]),
                       np.array([x2, x4, x3, x5, x6, x5, x7, x6, x8, x9, x8, x9]), temp)
                     for (x1, x2, x3, x4, x5, x6, x7, x8, x9) in perms])
        ztemps.append(ztemp)
    return dict(zip(temps, ztemps))


def y2row(y, width=8):
    """
    y: an integer in (0,...,(2**width)-1)
    """
    if not 0 <= y <= (2 ** width) - 1:
        raise ValueError(y)
    my_str = np.binary_repr(y, width=width)
    my_list = list(map(int, my_str))  # Python 3
    my_array = np.asarray(my_list)
    my_array[my_array == 0] = -1
    row = my_array
    return row


def asserts(one_dim_arrs: [np.array], non_zeros: [float]):
    for arr in one_dim_arrs:
        assert arr.ndim == 1, f"array must be 1xn but it is {arr.shape}"
    for num in non_zeros:
        assert num != 0, f"A number should have none zero value but it is {num}"


if __name__ == '__main__':
    a = np.arange(10) + 1
    b = np.arange(10) * 0.2
    c = np.array([[1, 2, 3]])
    print(f"Ex1: a={a}, G(a,1)={G(a, 1)}")
    print(f"Ex2: a={a}, b={b}, F(a,b,1)={F(a, b, 1)}")
    # print(f"Testing 1xn: {G(np.eye(2),1)}")
    # print(f"Testing temp: {F(a,b,0)}")
    print(f"Ex3 results: {ex3()}")
    print(f"Ex4 results: {ex4()}")
