from scipy.integrate import odeint

import matplotlib.pyplot as plt


class Model:

    def __init__(self, system, params, init_cond):

        self.__system = system
        self.__params = params
        self.__init_cond = init_cond

        self.__colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    def get_y(self, t):

        ret = odeint(self.__get_model, self.__init_cond, t)
        return ret.T

    def graph(self, t):

        y = self.get_y(t)

        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

        for i in range(len(y)):
            ax.plot(t, y[i], self.__colors[i % len(self.__colors)], alpha=0.5, lw=2)

        ax.set_ylim(0, 1.2)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.show()

    def __get_model(self, y, t):

        result = []
        cont = 0
        for equation in self.__system:
            right_member = 0
            for f in equation:
                right_member += self.__params[cont] * f(t, *y)
                cont += 1
            result.append(right_member)

        return result
