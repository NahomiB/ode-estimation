import numpy as np
import matplotlib.pyplot as plt
import random

# Define the SIR model
def get_SIR_model(y, t, a, b):
    S = y[0]
    I = y[1]
    R = y[2]
    dSdt = -a * S * I
    dIdt = a * S * I - b * I
    dRdt = b * I
    return [dSdt, dIdt, dRdt]


def noise(point, factor, additive):

    return max(0.0, point + random.uniform(-factor, factor) * (1 if additive else point))


# Graph a SIR model
def generate_points():

    # Define the parameters
    final_t = 100
    steps = 100

    a = 0.4
    b = 0.1
    S0 = 0.99
    I0 = 0.01
    R0 = 0.0

    n_factor = 0.1
    n_add = True

    # Define the time span
    t = np.linspace(0, final_t, steps)

    # Solve the ODEs
    from scipy.integrate import odeint
    y0 = [S0, I0, R0]
    ret = odeint(get_SIR_model, y0, t, args=(a, b))
    S, I, R = ret.T

    data = []
    with open('data/sir_model.csv', 'w') as f:
        f.write('t,S,I,R \n')
        for i in range(steps):
            points = [0.0, S0, I0, R0]
            if i > 0:
                points = [int(final_t / steps) * i, noise(S[i], n_factor, n_add), noise(I[i], n_factor, n_add), noise(R[i], n_factor, n_add)]
            f.write(f'{int(final_t / steps) * i},{points[1]},{points[2]},{points[3]} \n')
            data.append(points)

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S, 'b', alpha=0.2, lw=2, label='Susceptible')
    ax.plot(t, I, 'r', alpha=0.2, lw=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.2, lw=2, label='Recovered')

    D = np.array(data)
    ax.plot(D[:, 0], D[:, 1], 'bo')
    ax.plot(D[:, 0], D[:, 2], 'ro')
    ax.plot(D[:, 0], D[:, 3], 'go')

    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()

generate_points()
