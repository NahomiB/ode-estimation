import numpy as np
import matplotlib.pyplot as plt

# Define the SIR model
def get_SIR_model(y, t, a, b):
    S = y[0]
    I = y[1]
    R = y[2]
    dSdt = -a * S * I
    dIdt = a * S * I - b * I
    dRdt = b * I
    return [dSdt, dIdt, dRdt]


# Graph a SIR model
def generate_points():

    # Define the parameters
    final_t = 160
    steps = 160

    a = 0.3
    b = 0.2
    S0 = 0.99
    I0 = 0.01
    R0 = 0.0

    # Define the time span
    t = np.linspace(0, final_t, steps)

    # Solve the ODEs
    from scipy.integrate import odeint
    y0 = [S0, I0, R0]
    ret = odeint(get_SIR_model, y0, t, args=(a, b))
    S, I, R = ret.T

    with open('data/sir_model.csv', 'w') as f:
        f.write('t,S,I,R \n')
        for i in range(steps):
            if i % 5 == 0:
                f.write(f'{int(final_t / steps) * i},{S[i]},{I[i]},{R[i]} \n')

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S, 'b', alpha=0.2, lw=2, label='Susceptible')
    ax.plot(t, I, 'r', alpha=0.2, lw=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.2, lw=2, label='Recovered')

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
