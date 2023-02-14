import numpy as np

from modules.model import Model
from modules.points import generate_points, noise, smooth
from modules.estimator import Estimator

start_t = 0
final_t = 20
steps = 100

params = [0.4, 0.4, 0.1, 0.1]
restrictions = [[0, 1], [2, 3]]
y0 = [0.8, 0.2, 0.0]

n_factor = 0.2
n_add = False

seed = None

# Define the time span
t = np.linspace(start_t, final_t, steps)

f_11 = lambda t, S, I, R : -S * I
f_21 = lambda t, S, I, R : S * I
f_22 = lambda t, S, I, R : - I
f_31 = lambda t, S, I, R : I

system = [[f_11],
        [f_21, f_22],
        [f_31]]

model = Model(system, params, y0)
data = generate_points(model.get_Y(t), start_t, final_t, steps, n_factor, n_add, seed)
smooth_data = smooth(data, 0.9)

estimator = Estimator(system, restrictions, smooth_data, data, params)
estimator.graph(t)
