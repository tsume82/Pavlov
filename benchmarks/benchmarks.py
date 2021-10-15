import math


def rastrigin(c):
    return sum([x ** 2 - 10 * math.cos(2 * math.pi * x) + 10 for x in c])


def sphere(c):
    return sum([x ** 2 for x in c])


def ackley(c):
    dim = len(c)
    return (
        -20 * math.exp(-0.2 * math.sqrt(1.0 / dim * sum([x ** 2 for x in c])))
        - math.exp(1.0 / dim * sum([math.cos(2 * math.pi * x) for x in c]))
        + 20
        + math.e
    )


def griewank(c):
    prod = 1
    for i, x in enumerate(c):
        prod *= math.cos(x / math.sqrt(i + 1))
    return 1.0 / 4000.0 * sum([x ** 2 for x in c]) - prod + 1


def rosenbrock(c):
    total = 0
    for i in range(len(c) - 1):
        total += 100 * (c[i] ** 2 - c[i + 1]) ** 2 + (c[i] - 1) ** 2
    return total


def schwefel(c):
    return 418.9829 * len(c) - sum([x * math.sin(math.sqrt(abs(x))) for x in c])


all = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "griewank": griewank,
    "rosenbrock": rosenbrock,
    "schwefel": schwefel,
}
