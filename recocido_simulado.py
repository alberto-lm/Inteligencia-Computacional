import math
import numpy as np
import matplotlib.pyplot as plt   

class RecSim(object):
    def __init__(self, coords=-1, alpha=-1, beta=-1, M=-1, R=-1, stopping_iteration=-1):
        self.coords = np.random.randint(0, 20, size=(10,2)) if coords == -1 else coords 
        self.alpha = 0.9 if alpha == -1 else alpha
        self.beta = 1.5 if beta == -1 else beta
        self.N = len(self.coords)
        self.M = 50 if M == -1 else M
        self.R = 0.95 if R == -1 else R
        self.stopping_iteration = 1000 if stopping_iteration == -1 else stopping_iteration
        self.iteration = 1
        self.current_solution, self.current_distance, self.best_solution, self.shortest_distance, self.initial_solution = self.init_paths()
        self.distances_list = []
        self.T = self.init_temp()

    def init_paths(self):
        cur_sol = []
        nodes = [i for i in range(self.N)]
        while len(nodes) > 0:
            i = np.random.randint(0, len(nodes))
            cur_sol.append(nodes[i])
            nodes.pop(i)
        best_sol = cur_sol.copy()
        cur_dist = self.solution_distance(cur_sol)
        best_dist = self.solution_distance(best_sol)
        return cur_sol.copy(), cur_dist, best_sol.copy(), best_dist, cur_sol.copy()

    def two_points_distance(self, a, b):
        coord_a, coord_b = self.coords[a], self.coords[b]
        return math.sqrt((coord_a[0] - coord_b[0]) ** 2 + (coord_a[1] - coord_b[1]) ** 2)

    def solution_distance(self, solution):
        distance = 0
        for i in range(self.N):
            distance += self.two_points_distance(solution[i % self.N], solution[(i + 1) % self.N])
        return distance

    def p_accept(self, candidate_distance):
        return math.exp(-abs(candidate_distance - self.current_distance) / self.T)

    def generate_mutation(self):
        parent = self.current_solution.copy()
        i = np.random.randint(0, int(self.N/2))
        j = np.random.randint(int(self.N/2), self.N)
        parent[i], parent[j] = parent[j], parent[i]
        return parent

    def cadena_markov(self):
        accepted = 0
        local_distances_list = []
        for i in range(self.M):
            candidate = self.generate_mutation()
            candidate_distance = self.solution_distance(candidate)
            if candidate_distance < self.current_distance:
                self.current_distance, self.current_solution = candidate_distance, candidate.copy()
                accepted += 1
                if candidate_distance < self.shortest_distance:
                    self.shortest_distance, self.best_solution = candidate_distance, candidate.copy()
            elif np.random.uniform() < self.p_accept(candidate_distance):
                accepted += 1
                self.current_distance, self.current_solution = candidate_distance, candidate
            local_distances_list.append(self.current_distance)
        return accepted/self.M, local_distances_list

    def init_temp(self):
        self.T = 0.1
        r = 0
        while r < self.R:
            r, _ = self.cadena_markov()
            self.T *= self.beta
        print(self.T)
        return self.T
    
    def recocido(self):
        self.current_solution, self.current_distance, self.best_solution, self.shortest_distance, self.initial_solution = self.init_paths()
        while self.iteration < self.stopping_iteration:
            _, local_distances_list = self.cadena_markov()
            self.distances_list.append(local_distances_list)
            self.T *= self.alpha
            self.iteration += 1

    def plot_learning(self):
        evaluaciones = np.asarray(self.distances_list)
        avg = np.average(evaluaciones, axis=1)
        std = np.std(evaluaciones, axis=1)
        arriba = avg + std
        abajo = avg - std
        plt.plot(arriba, color='r')
        plt.plot(avg, color='b')
        plt.plot(abajo, color='g')
        plt.ylabel("Distance")
        plt.xlabel("Iteration")
        plt.show()

    def plot_paths(self):
        initial = self.initial_solution
        final = self.best_solution
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        i_path = list(zip(initial[1:], initial[:-1])) + [(initial[0], initial[-1])]
        f_path = list(zip(final[1:], final[:-1])) + [(final[0], final[-1])]
        for j in range(self.N):
            i_first, i_second = i_path[j]
            f_first, f_second = f_path[j]
            ax1.plot(self.coords[initial[j]][0], self.coords[initial[j]][1], 'ro')
            ax2.plot(self.coords[final[j]][0], self.coords[final[j]][1], 'ro')
            ax1.plot([self.coords[i_first][0], self.coords[i_second][0]], [self.coords[i_first][1], self.coords[i_second][1]], 'b')
            ax2.plot([self.coords[f_first][0], self.coords[f_second][0]], [self.coords[f_first][1], self.coords[f_second][1]], 'b')
        plt.show()
        
            

if __name__ == "__main__":
    rs = RecSim(stopping_iteration=100)
    rs.recocido()
    rs.plot_learning()
    rs.plot_paths()
