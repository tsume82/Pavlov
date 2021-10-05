import math

def rastrigin(population):
    return sum([x ** 2 - 10 * math.cos(2 * math.pi * x) + 10 for x in population])

def sphere(population):
    return sum([x ** 2 for x in population])