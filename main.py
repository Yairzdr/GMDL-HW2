import numpy as np
import itertools
import functools
import operator
import matplotlib.pyplot as plt


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


def ex5():
    length = 2
    ys = np.arange(4)
    perms = list(itertools.product(ys, repeat=length))
    temps = [1, 1.5, 2]
    ztemps = []
    for temp in temps:
        g, f = adjustFunctions(temp, width=2)  # note on page 4
        ztemp = sum([(g(y1) * g(y2) * f(y1, y2))
                     for (y1, y2) in perms])
        ztemps.append(ztemp)
    return dict(zip(temps, ztemps))


def ex6():
    length = 3
    ys = np.arange(8)
    perms = list(itertools.product(ys, repeat=length))
    temps = [1, 1.5, 2]
    ztemps = []
    for temp in temps:
        g, f = adjustFunctions(temp, width=3)  # note on page 5
        ztemp = sum([(g(y1) * g(y2) * g(y3) * f(y1, y2) * f(y2, y3))
                     for (y1, y2, y3) in perms])
        ztemps.append(ztemp)
    return dict(zip(temps, ztemps))


def get_ts_ps(temps, length):
    length_square = 2 ** length
    k = np.arange(length_square)
    results = dict(zip(temps, [{"T": None, "p": None, "Es": None} for i in temps]))
    for temp in temps:
        g, f = adjustFunctions(temp, width=length)
        Ts = np.array([[0.] * length_square] * length)
        ps = np.array([np.zeros((length_square, length_square))] * length)
        for i in range(length):
            # first
            if i == 0:
                # every T1(y_a) = sigma(G(y)F(y,y_a) for 0<=y<=length_square
                t = lambda y2: sum(np.array(list(map(
                    lambda y1: g(y1) * f(y1, y2), k))))
                apply_sum = np.array(list(map(t, k)))
                # set the corresponding y in the array
                for j in range(length_square):
                    Ts[i][j] = apply_sum[j]
            # all between
            elif i < length - 1:
                pre_t = Ts[i - 1]  # holds the previous t function value
                t = lambda yk_1: sum(np.array(list(map(
                    lambda yk: pre_t[yk] * g(yk) * f(yk, yk_1), k))))
                apply_sum = np.array(list(map(t, k)))
                for j in range(length_square):
                    Ts[i][j] = apply_sum[j]
            # last
            else:
                t = sum(np.array(list(map(
                    lambda yk: Ts[i - 1][yk] * g(yk), k))))
                Ts[i] = t

        ztemp = Ts[length - 1][0]

        for i in range(length - 1, -1, -1):
            # last
            if i == length - 1:
                #  TODO - fix a bug > 1.
                p = lambda y: (Ts[i - 1][y] * g(y)) / ztemp
                py = np.array(list(map(p, k)))
                ps[i] = py
            # all between
            elif i > 0:
                p = lambda yk, yk_1: (Ts[i - 1][yk] * g(yk) * f(yk, yk_1)) / (Ts[i][yk_1])
                py = np.array(list(map(
                    lambda yk_1:
                    np.array(list(map(
                        lambda y: p(y, yk_1), k))), k)))
                ps[i] = py
            # last
            else:
                p = lambda y1, y2: (g(y1) * f(y1, y2)) / (Ts[i][y2])
                py = np.array(list(map(
                    lambda y2:
                    np.array(list(map(
                        lambda y1: p(y1, y2), k))), k)))
                ps[i] = py
        results[temp]["T"] = Ts
        results[temp]["p"] = ps
    return results


def sampler(ps, length, n_samples):
    results = np.array([np.ones((length, length))] * n_samples)
    rand = lambda p: np.random.choice(np.arange(2 ** length), p=p)
    currY = 0  # empty init
    for i in range(n_samples):
        for j in range(length - 1, -1, -1):
            if j == length - 1:
                #  TODO fix prob sum > 1
                currY = rand(ps[j][0])  # to get the 1xlength
                results[i][j] = y2row(currY, length)
            else:
                currY = rand(ps[j][currY])
                results[i][j] = y2row(currY, length)
    return results


def ex7(temps=[1, 1.5, 2], length=8):
    n_samples = 10
    d = get_ts_ps(temps, length)
    fig, axs = plt.subplots(len(temps), n_samples, dpi=150)
    fig.suptitle("Samples per temp")
    for i, temp in enumerate(d):
        axs[i, 0].set_ylabel(f'temp={temp}')
        for j, sample in enumerate(sampler(d[temp]["p"], length, n_samples)):
            axs[i, j].imshow(sample, interpolation='None')
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()


def ex8(temps=[1, 1.5, 2], length=8):
    n_samples = 10000
    d = get_ts_ps(temps, length)
    for i, temp in enumerate(d):
        Es = {"11": 0, "18": 0}
        for _ in range(n_samples):
            sample = sampler(d[temp]["p"], length, 1)[0]
            Es["11"] += sample[0][0] * sample[1][1]
            Es["18"] += sample[0][0] * sample[length - 1][length - 1]
        Es["11"] /= n_samples
        Es["18"] /= n_samples
        d[temp]["Es"] = Es
        print(d[temp]["Es"])


def print_results(results):
    for temp in results:
        print(f"\ntemp={temp}:")
        for i, arr in enumerate(results[temp]["T"]):
            print(f"T{i + 1} = {arr}")
            print(f"p{i + 1} = {results[temp]['p'][i].shape}")
            print('-' * 25)


def adjustFunctions(temp, width):
    """
    Given temp and a width is cast all y occurrences using y2row,
    returns corresponding G, and F
    """
    return lambda y: G(y2row(y, width), temp), lambda y1, y2: F(y2row(y1, width), y2row(y2, width), temp)


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
    # print(f"Ex1: a={a}, G(a,1)={G(a, 1)}")
    # print(f"Ex2: a={a}, b={b}, F(a,b,1)={F(a, b, 1)}")
    # print(f"Testing 1xn: {G(np.eye(2),1)}")
    # print(f"Testing temp: {F(a,b,0)}")
    # for ex in [ex3, ex4, ex5, ex6,ex7]:
    #     print(f"{ex.__name__} results: {ex()}")
    # ex7()
    ex8()
