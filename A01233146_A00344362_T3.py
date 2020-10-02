import math
import numpy as np
import matplotlib.pyplot as plt

class Particula:

    def __init__(self, x=-1, v=-1, alpha=-1, beta=-1):
        self.x = np.random.uniform(0, 1, size=3) * 10 if x == -1 else x
        self.l = self.x
        self.v = np.zeros(3) if v == -1 else v
        self.alpha = 2 if alpha == -1 else alpha
        self.beta = 2 if beta == -1 else beta

    def next_state(self, g):
        self.v = self.get_next_v(g)
        self.x = self.x + self.v

    def get_next_v(self, g):
        ep_1 = np.random.uniform(0, 1, size=3)
        ep_2 = np.random.uniform(0, 1, size=3)
        v_1 = self.v + self.alpha * ep_1 * (g - self.x) + self.beta * ep_2 * (self.l - self.x)
        mag = np.linalg.norm(v_1)
        if mag == 0:
            return v_1
        return v_1 / mag

class PSO:

    def __init__(self, lut, n=-1, max_iterations=-1):
        self.lut = lut
        self.n = 10 if n == -1 else n
        self.particulas = [Particula() for i in range(self.n)]          
        self.g = self.get_best_local()
        self.g_list = []
        self.max_iterations = 100 if max_iterations == -1 else max_iterations
        self.iteration = 0

    def get_best_local(self):
        err = math.inf
        best_local = [0, 0, 0]
        for p in self.particulas:
            aux = self.calc_error(p.l)
            if aux < err:
                err = aux
                best_local = p.l
        return best_local

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
            for p in self.particulas:
                p.next_state(self.g)
                err_x = self.calc_error(p.x)
                err_l = self.calc_error(p.l)
                if err_x < err_l:
                    p.l = p.x
            self.g = self.get_best_local()
            self.g_list.append(self.calc_error(self.g))
            self.iteration += 1
        print(f'error = {self.calc_error(self.g)}')
        return self.g

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
    pso = PSO(lut=values, n=100, max_iterations=50)
    coefficients = pso.calc_coefficients()
    print(coefficients)
    pso.plot_g()
