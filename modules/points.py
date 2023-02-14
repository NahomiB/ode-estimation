import numpy as np
import random
import sys
from scipy.interpolate import UnivariateSpline


def noise(point, factor, additive):

    return max(0.0, point + random.uniform(-factor, factor) * (1.0 if additive else point))


def generate_points(Y, start_t, final_t, steps, n_factor, n_add, seed = None):

    new_seed = random.randrange(sys.maxsize) if seed == None else seed
    random.seed(new_seed)
    print('Noise seed: ' + str(new_seed))

    data = []
    for i in range(steps):
        points = [start_t + i * (final_t - start_t) / steps]
        for j in range(len(Y)):
            points.append(noise(Y[j][i], n_factor, n_add) if i > 0 else Y[j][i])
        data.append(points)

    D = np.array(data)
    return D

def smooth(data, factor):

    # Smooth the data with a smoothing spline
    x = data[:, 0]
    new_data = []
    new_data.append(x)
    for k in range(1, len(data[0])):
        y = data[:, k]
        spl = UnivariateSpline(x, y)
        spl.set_smoothing_factor(factor)
        y_smooth = spl(x)

        for i in range(len(y_smooth)):
            y_smooth[i] = max(0.0, y_smooth[i])
        
        new_data.append(y_smooth)

    D = np.array(new_data)


    return D.transpose()
    
