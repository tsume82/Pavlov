from math import sin, cos, pi, sqrt, e, exp

def rastrigin(c):
    return sum([x ** 2 - 10 * cos(2 * pi * x) + 10 for x in c])


def sphere(c):
    return sum([x ** 2 for x in c])

def ackley(c):
    dim = len(c)
    return (
        -20 * exp(-0.2 * sqrt(1.0 / dim * sum([x ** 2 for x in c])))
        - exp(1.0 / dim * sum([cos(2 * pi * x) for x in c]))
        + 20
        + e
    )

def griewank(c):
    prod = 1
    for i, x in enumerate(c):
        prod *= cos(x / sqrt(i + 1))
    return 1.0 / 4000.0 * sum([x ** 2 for x in c]) - prod + 1

def rosenbrock(c):
    total = 0
    for i in range(len(c) - 1):
        total += 100 * (c[i] ** 2 - c[i + 1]) ** 2 + (c[i] - 1) ** 2
    return total

def schwefel(c):
    return 418.9829 * len(c) - sum([x * sin(sqrt(abs(x))) for x in c])

all = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "griewank": griewank,
    "rosenbrock": rosenbrock,
    "schwefel": schwefel,
}
