import math
import numpy as np
import matplotlib.pyplot as plt

class FFA:

    def __init__(self, x=-1, beta=-1, gamma=-1):
        self.x = np.random.uniform(0, 1, size=3) * 10 if x == -1 else x
        self.gamma = 1 if gamma == -1 else gamma
        self.beta = 1 if beta == -1 else beta

    def next_state(self, j):
        r = self.get_distance(j)
        alpha = np.random.uniform(0, 1)
        epsilon = np.random.uniform(0, 1, size=3) - 0.5
        aux = [0, 0, 0]
        try:
            aux = self.beta * np.exp(-1 * self.gamma * r ** 2) * (j - self.x)
        except:
            pass
        self.x = self.x + aux + alpha * epsilon

    def get_distance(self, j):
        return np.linalg.norm(j - self.x)

class FFA_Alg:

    def __init__(self, lut, n=-1, max_iterations=-1):
        self.lut = lut
        self.n = 10 if n == -1 else n
        self.ffas = [FFA() for i in range(self.n)]          
        self.g_list = []
        self.max_iterations = 100 if max_iterations == -1 else max_iterations
        self.iteration = 0

    def get_ffa(self):
        err = math.inf
        best_local = [0, 0, 0]
        for f in self.ffas:
            aux = self.calc_error(f.x)
            if aux < err:
                err = aux
                best_local = f.x
        return best_local, err

    def calc_error(self, particula):
        err = 0.0
        for x in self.lut:
            y = self.evaluate(particula, x)
            try:
                err += (y - self.lut[x])**2
            except:
                return math.inf
        return err / len(self.lut)

    def evaluate(self, particula, x):
        a_1, a_2, a_3 = particula
        try:
            return a_1 * x ** 2 - a_2 ** (math.exp(-1 * a_3 * x) / 2)
        except:
            return math.inf

    def calc_coefficients(self):
        while self.iteration < self.max_iterations:
            for i in self.ffas:
                for j in self.ffas:
                    err_i = self.calc_error(i.x)
                    err_j = self.calc_error(j.x)
                    if err_j < err_i:
                        i.next_state(j.x)
            _, err = self.get_ffa()
            self.g_list.append(err)
            self.iteration += 1
        g, err = self.get_ffa()
        print(f'error = {err}')
        print(f'solution = {g}')
        return g

    def plot_g(self):
        err = np.asarray(self.g_list)
        plt.plot(err, color='g')
        plt.ylabel("Error")
        plt.xlabel("Iteration")
        plt.show()

if __name__ == "__main__":
    values = {
        0: -3.271085446759225,
        1: 2.9946329498568343,
        2: 14.999975823810116,
        3: 34.999999890807594,
        4: 62.99999999950683,
        5: 98.99999999999777,
        6: 143,
        7: 195,
        8: 255,
        9: 323
    }
    np.seterr('raise')
    ffa = FFA_Alg(lut=values, n=100, max_iterations=100)
    coefficients = ffa.calc_coefficients()
    print(coefficients)
    ffa.plot_g()
