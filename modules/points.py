import numpy as np
import random
import sys

def noise(point, factor, additive):

    return max(0.0, point + random.uniform(-factor, factor) * (1 if additive else point))


def generate_points(Y, final_t, steps, n_factor, n_add, seed = None):

    new_seed = random.randrange(sys.maxsize) if seed == None else seed
    random.seed(new_seed)
    print('Noise seed: ' + str(new_seed))

    data = []
    for i in range(steps):
        points = [i * final_t / steps]
        for j in range(len(Y)):
            points.append(noise(Y[j][i], n_factor, n_add) if i > 0 else Y[j][i])
        data.append(points)

    D = np.array(data)
    return D