import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

# Create a class for the model
class Estimator:
    """
    A class used to represent a system of linear ordinary differential equations

    Attributes
    ----------
    params -> list
        Return the estimated parameters of the system in order

    Methods
    -------
    graph(start_t, final_t, steps):
        Plot the estimated system of equations and the data
    """

    def __init__(self, system : list, restrictions : list, data : list, original_params = []):
        """
        Parameters
        ----------
        system : list
            A list of lists of functions that represent the system of equations
        restrictions: list
            A list of pairs, where a pair represents the index of the parameters that must be equal
        data: list
            A set of points that must be approximated
        """
        
        self.__colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']        

        self.__system = system
        self.__restrictions = restrictions
        self.__D = data

        self.__params = self.__obtain_params()
        self.params = self.__params[:len(self.__params) - len(self.__restrictions)]

        self.__original_params = original_params
        self.__original_flag = False


    def params(self) -> list:
        """
        Return the estimated parameters of the system in order
        """

        return self.__params[:len(self.__params) - len(self.__restrictions)]


    def graph(self, t):
        """
        Plot the estimated system of equations and the data

        Parameters
        ----------
        start_t : float
            The initial time
        final_t : float
            The final time
        steps : int
            The number of steps
        """

        # Solve the ODEs
        from scipy.integrate import odeint
        y0 = self.__D[0][1:]
        ret = odeint(self.__get_model, y0, t)
        Y = ret.T

        Y_original = []
        if self.__original_params != []:
            self.__original_flag = True
            ret_original = odeint(self.__get_model, y0, t)
            Y_original = ret_original.T
            self.__original_flag = False

        # Plot the aproximated system
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        for i in range(len(Y)):
            ax.plot(t, Y[i], self.__colors[i % len(self.__colors)], alpha=0.5, lw=2, label='Approx')

        # Plot the original system
        for i in range(len(Y_original)):
            ax.plot(t, Y_original[i], self.__colors[i % len(self.__colors)], alpha=0.5, lw=2, linestyle='dashed', label='Original')

        # Plot the data as points
        for i in range(len(Y)):
            ax.plot(self.__D[:, 0], self.__D[:, i + 1], self.__colors[i % len(self.__colors)] + 'o', label='Data')

        p = self.params
        params_text = ''
        for i in range(len(p)):
            if i > 0: params_text += '\n'
            o_param = '- Orig: ' + str(self.__original_params[i]) if len(self.__original_params) > 0 else ''
            params_text += f'a{str(i + 1)} = {str(round(p[i], 5))} {o_param}'

        at = AnchoredText(
            params_text, prop=dict(size=10), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

        ax.set_ylim(0,1.2)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.show()


    def __obtain_params(self):
        
        mat_size = len(self.__restrictions)
        for eq in self.__system:
            mat_size += len(eq)    

        # Create normal matrix upper part
        normal_mat = np.zeros((mat_size, mat_size))

        current_row = 0
        for i in range(len(self.__system)): # For each equation...
            for j in range(len(self.__system[i])): # ...and for each parameter...
                for k in range(j, len(self.__system[i])): # ...and for each parameter again...

                    normal_mat[current_row + j][current_row + k] = self.__scalar_product(self.__system[i][j], self.__system[i][k], self.__D)
                    normal_mat[current_row + k][current_row + j] = normal_mat[current_row + j][current_row + k]
            current_row += len(self.__system[i])
        
        # Embed restrictions into normal matrix
        row = mat_size - len(self.__restrictions)
        for r in self.__restrictions:
            normal_mat[row][r[0]] = 1
            normal_mat[row][r[1]] = -1
            normal_mat[r[0]][row] = 1
            normal_mat[r[1]][row] = -1
            row += 1

        # Create right side vector for the linear equations system
        b = np.zeros(mat_size)

        cont = 0
        for eq in range(len(self.__system)):
            for param in range(len(self.__system[eq])):
                b[cont] += self.__scalar_product_deriv(eq + 1, self.__system[eq][param], self.__D)
                cont += 1

        # Solve the system
        x = np.linalg.solve(normal_mat, b)
        return x


    def __get_model(self, y, t):

        result = []
        cont = 0
        for equation in self.__system:
            right_member = 0
            for f in equation:
                right_member += (self.__params[cont] if self.__original_flag == False else self.__original_params[cont]) * f(t, *y)
                cont += 1
            result.append(right_member)

        return result


    def __scalar_product(self, f, g, D):

        sum = 0
        for i in range(len(D)):
            sum += f(D[i][0], D[i][1], D[i][2], D[i][3]) * g(D[i][0], D[i][1], D[i][2], D[i][3])
        return sum


    def __scalar_product_deriv(self, col, f, D):

        sum = 0
        for i in range(len(D) - 1):
            m = (D[i + 1][col] - D[i][col]) / (D[i + 1][0] - D[i][0])
            sum += f(D[i][0], D[i][1], D[i][2], D[i][3]) * m
        return sum

