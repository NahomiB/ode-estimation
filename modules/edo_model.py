import numpy as np
import matplotlib.pyplot as plt

# Create a class for the model
class EDOModel:


    def __init__(self, system, restrictions, data):

        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']        

        self.system = system
        self.restrictions = restrictions
        self.D = data

        self.params = self.__obtain_params()

    def __obtain_params(self):
        
        mat_size = len(self.restrictions)
        for eq in self.system:
            mat_size += len(eq)    

        # Create normal matrix upper part
        normal_mat = np.zeros((mat_size, mat_size))

        current_row = 0
        for i in range(len(self.system)): # For each equation...
            for j in range(len(self.system[i])): # ...and for each parameter...
                for k in range(j, len(self.system[i])): # ...and for each parameter again...

                    normal_mat[current_row + j][current_row + k] = self.__scalar_product(self.system[i][j], self.system[i][k], self.D)
                    normal_mat[current_row + k][current_row + j] = normal_mat[current_row + j][current_row + k]
            current_row += len(self.system[i])
        
        # Embed restrictions into normal matrix
        row = mat_size - len(self.restrictions)
        for r in self.restrictions:
            normal_mat[row][r[0]] = 1
            normal_mat[row][r[1]] = -1
            normal_mat[r[0]][row] = 1
            normal_mat[r[1]][row] = -1
            row += 1

        # Create right side vector for the linear equations system
        b = np.zeros(6)

        cont = 0
        for eq in range(len(self.system)):
            for param in range(len(self.system[eq])):
                b[cont] += self.__scalar_product_deriv(eq + 1, self.system[eq][param], self.D)
                cont += 1

        # Solve the system
        x = np.linalg.solve(normal_mat, b)
        return x


    def get_model(self, y, t):

        result = []
        cont = 0
        for equation in self.system:
            right_member = 0
            for f in equation:
                right_member += self.params[cont] * f(t, *y)
                cont += 1
            result.append(right_member)

        return result


    def graph(self):

        t = np.linspace(0, len(self.D), len(self.D))
        
        # Solve the ODEs
        from scipy.integrate import odeint
        y0 = self.D[0][1:]
        ret = odeint(self.get_model, y0, t)
        Y = ret.T

        # Plot the data on three separate curves for S(t), I(t) and R(t)
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        for i in range(len(Y)):
            ax.plot(t, Y[i], self.colors[i % len(self.colors)], alpha=0.5, lw=2, label='y(t)')

        # Get a subset of the data to plot
        data_subset = self.D[::10]

        # Plot the data as points
        for i in range(len(Y)):
            ax.plot(data_subset[:, 0], data_subset[:, i + 1], self.colors[i % len(self.colors)] + 'o')

        ax.set_ylim(0,1.2)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.show()


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

