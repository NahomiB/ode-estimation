import numpy as np
import random

def noise(point, factor, additive):

    return max(0.0, point + random.uniform(-factor, factor) * (1 if additive else point))


def generate_points(Y, steps, n_factor, n_add):

    data = []
    for i in range(steps):
        points = [i]
        for j in range(len(Y)):
            points.append(noise(Y[j][i], n_factor, n_add) if i > 0 else Y[j][i])
        data.append(points)

    D = np.array(data)
    return D