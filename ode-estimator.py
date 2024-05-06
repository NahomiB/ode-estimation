import numpy as np

from modules.estimator import Estimator
from modules.model import Model
from modules.points import generate_points, smooth


def estimate(system, params, y0, restrictions, start_t=0, final_t=20, steps=100, n_factor=0.2,
             smooth_factor=0.9, n_add=False, seed=None):
    # Define the time span
    t = np.linspace(start_t, final_t, steps)

    model = Model(system, params, y0)
    data = generate_points(model.get_y(t), start_t, final_t, steps, n_factor, n_add, seed)
    smooth_data = smooth(data, smooth_factor)

    estimator = Estimator(system, restrictions, smooth_data, data, params)
    return estimator
