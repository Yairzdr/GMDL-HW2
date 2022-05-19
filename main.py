import numpy as np
import itertools
import functools
import operator
import matplotlib.pyplot as plt


def G(row_s: np.array, temp: float): return np.exp((1 / temp) * row_s[:-1] @ row_s[1:])


def F(row_s: np.array, row_t: np.array, temp: float): return np.exp((1 / temp) * (row_s @ row_t))


def sq(x): return x ** 2


def create_dict(temps, *vals):
    return dict(zip(temps, [dict(zip(vals, [None] * len(vals))) for _ in temps]))


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


def adjustFunctions(temp, width):
    """
    Given temp and a width is cast all y occurrences using y2row,
    returns corresponding G, and F
    """
    return lambda y: G(y2row(y, width), temp), lambda y1, y2: F(y2row(y1, width), y2row(y2, width), temp)


def ex3(temps=[1, 1.5, 2], n=2):
    perms = list(itertools.product([-1, 1], repeat=sq(n)))  # [(-1,-1,-1,-1),(1,-1,-1,-1),(-1,1,-1,-1)...]
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


def ex4(temps=[1, 1.5, 2], n=3):
    perms = list(itertools.product([-1, 1], repeat=sq(n)))
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


def ex5(temps=[1, 1.5, 2], length=2):
    ys = np.arange(2 ** length)
    perms = list(itertools.product(ys, repeat=length))
    ztemps = []
    for temp in temps:
        g, f = adjustFunctions(temp, width=2)  # note on page 4
        ztemp = sum([(g(y1) * g(y2) * f(y1, y2))
                     for (y1, y2) in perms])
        ztemps.append(ztemp)
    return dict(zip(temps, ztemps))


def ex6(temps=[1, 1.5, 2], length=3):
    ys = np.arange(2 ** length)
    perms = list(itertools.product(ys, repeat=length))
    ztemps = []
    for temp in temps:
        g, f = adjustFunctions(temp, width=3)  # note on page 5
        ztemp = sum([(g(y1) * g(y2) * g(y3) * f(y1, y2) * f(y2, y3))
                     for (y1, y2, y3) in perms])
        ztemps.append(ztemp)
    return dict(zip(temps, ztemps))


def get_ts_ps(temps, length):
    """
    Returns the Ts and Ps
    """
    length_square = 2 ** length
    k = np.arange(length_square)
    results = create_dict(temps, "T", "p", "Es")
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


def random_sampler(ps, length, n_samples):
    results = np.array([np.ones((length, length))] * n_samples)
    rand = lambda p: np.random.choice(np.arange(2 ** length), p=p)
    currY = 0  # empty init
    for i in range(n_samples):
        for j in range(length - 1, -1, -1):
            if j == length - 1:
                currY = rand(ps[j][0])  # to get the (1 x length) shape
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
        for j, sample in enumerate(random_sampler(d[temp]["p"], length, n_samples)):
            axs[i, j].imshow(sample, interpolation='None')
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()
    return "7 Done"


def ex8(temps=[1, 1.5, 2], length=8):
    n_samples = 10000
    results = create_dict(temps, "Es")
    d = get_ts_ps(temps, length)
    for i, temp in enumerate(results):
        Es = {"12": 0, "18": 0}
        for _ in range(n_samples):
            sample = random_sampler(d[temp]["p"], length, 1)[0]
            Es["12"] += sample[0][0] * sample[1][1]
            Es["18"] += sample[0][0] * sample[length - 1][length - 1]
        Es["12"] /= n_samples
        Es["18"] /= n_samples
        results[temp]["Es"] = Es
    return results


def sampler_ex9(length=8, inner_lattice=None):
    if inner_lattice is None:
        inner_lattice = np.random.randint(low=0, high=2, size=(length, length)) * 2 - 1
    return np.pad(inner_lattice, [1, 1], mode='constant')


def sweep(lattice, temp, length=8, posterior=False, sigma=2, y=None):
    b = 1 / temp
    add_guassian = lambda pos: - ((1 / (2 * sq(sigma))) * sq(y[i][j] + (- 1 if pos else 1)))
    for i in range(1, length + 1):  # because the lattice is padded
        for j in range(1, length + 1):
            sum_ni = lattice[i - 1][j] + lattice[i + 1][j] + lattice[i][j - 1] + lattice[i][j + 1]
            if not posterior:
                ppos = np.exp(b * sum_ni)
                pneg = np.exp(-b * sum_ni)
            else:
                ppos = np.exp(b * sum_ni + add_guassian(True))
                pneg = np.exp(-b * sum_ni + add_guassian(False))
            ppos, pneg = ppos / (ppos + pneg), pneg / (ppos + pneg)
            lattice[i][j] = np.random.choice([1, -1], p=[ppos, pneg])
    return lattice


def ICM(y, temp, size=100, sigma=2, max_iterations=50):
    b = 1 / temp
    lattice = sampler_ex9(size)
    add_guassian = lambda pos: - ((1 / (2 * sq(sigma))) * sq(y[i][j] + (- 1 if pos else 1)))
    flipped = True
    k = 0
    while flipped and k < max_iterations:
        k += 1
        for i in range(1, size + 1):  # because the lattice is padded
            for j in range(1, size + 1):
                sum_ni = lattice[i - 1][j] + lattice[i + 1][j] + lattice[i][j - 1] + lattice[i][j + 1]
                ppos = np.exp(b * sum_ni + add_guassian(True))
                pneg = np.exp(-b * sum_ni + add_guassian(False))
                if ppos > pneg:
                    if lattice[i][j] == -1:
                        flipped = True
                        lattice[i][j] = 1
                else:
                    if lattice[i][j] == 1:
                        flipped = True
                        lattice[i][j] = -1
    return lattice


def ex9(temps=[1, 1.5, 2], length=8):
    results = create_dict(temps, "m1", "m2")
    for temp in temps:
        results[temp]["m1"] = method1(temp, length=length)
        results[temp]["m2"] = method2(temp, length=length)
    return results


def method1(temp, n_samples=10000, n_sweeps=25, length=8):
    E12 = 0
    E18 = 0
    for s in range(n_samples):
        sample = sampler_ex9(length)
        print(f'm1 s:{s}') if s % 1000 == 0 else None
        for _ in range(n_sweeps):
            sample = sweep(sample, temp, length)
        E12 += sample[1][1] * sample[2, 2]
        E18 += sample[1][1] * sample[length, length]
    return (E12 / n_samples), (E18 / n_samples)


def method2(temp, n_sweeps=25000, length=8):
    E12 = 0
    E18 = 0
    s_start_sampling = 100
    sample = sampler_ex9(length)
    for s in range(n_sweeps):
        print(f'm2 s:{s}') if s % 1000 == 0 else None
        sample = sweep(sample, temp, length)
        if s >= s_start_sampling:
            E12 += sample[1][1] * sample[2, 2]
            E18 += sample[1][1] * sample[length, length]
    return (E12 / (n_sweeps - s_start_sampling)), (E18 / (n_sweeps - s_start_sampling))


def ex10(temps=[1, 1.5, 2], size=100):
    n_sweeps = 50
    sigma = 2
    types = ["X", "Y", "Posterior", "ICM", "MLE"]
    results = create_dict(temps, *types)
    for temp in temps:
        # 1) X - Gibbes sampling
        sample = sampler_ex9(size)
        for _ in range(n_sweeps):
            sample = sweep(sample, temp, size)

        # 2) Y - Gaussian noise
        eta = sampler_ex9(size, inner_lattice=(sigma * np.random.standard_normal(size=(size, size))))
        y = sample + eta

        # 3) Posterior
        posterior_sample = sampler_ex9(size)
        for _ in range(n_sweeps):
            posterior_sample = sweep(sample, temp, size, posterior=True, sigma=2, y=y)

        # 4) ICM
        icm = ICM(y, temp, size, sigma, max_iterations=50)

        # 5) MLE
        mle = np.sign(y)
        results[temp] = dict(zip(types, [sample, y, posterior_sample, icm, mle]))

    fig, axs = plt.subplots(3, 5, dpi=150)
    fig.suptitle("Binary Image Restoration")
    for i, temp in enumerate(results):
        axs[i, 0].set_ylabel(f'temp={temp}')
        for j, type in enumerate(results[temp]):
            if j == 1:
                axs[i, j].imshow(results[temp][type], interpolation='None')
            else:
                axs[i, j].imshow(results[temp][type], vmin=-1, vmax=1, interpolation='None')
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
            axs[i, j].title.set_text(type)
    plt.show()
    return results


if __name__ == '__main__':
    for ex in [ex3, ex4, ex5, ex6, ex7, ex8, ex9, ex10]:
        print(f"\n{ex.__name__}\nresults: {ex()}\n\n")
