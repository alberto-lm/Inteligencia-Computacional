from deap import base, creator, tools, algorithms
import numpy as np
import random
import argparse
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd


MAX_WEIGHT = 165

lut = {
    "1": {
        "w":23,
        "p":92
    },
    "2": {
        "w":31,
        "p":57
    },
    "3": {
        "w":29,
        "p":49
    },
    "4": {
        "w":44,
        "p":68
    },
    "5": {
        "w":53,
        "p":60
    },
    "6": {
        "w":38,
        "p":43
    },
    "7": {
        "w":63,
        "p":67
    },
    "8": {
        "w":85,
        "p":847
    },
    "9": {
        "w":89,
        "p":87
    },
    "10": {
        "w":82,
        "p":72
    }
}

def func_eval(solution, lut, max_weight):
    weight = 0
    profit = 0
    for idx, obj in enumerate(solution):
        if obj == 1:
            weight += lut[str(idx+1)]['w']
            profit += lut[str(idx+1)]['p']
    if weight > max_weight:
        return np.max(profit - (weight - max_weight)*4, 0),
    return profit,


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("select", tools.selRoulette)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.5)
toolbox.register("evaluate", func_eval, lut=lut, max_weight=MAX_WEIGHT)

toolbox.register("attribute", random.randint, a=0, b=1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(n=10)
hof = tools.HallOfFame(3)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("std", np.std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True)
    args = parser.parse_args()
    algorithm = args.algorithm
    df = pd.DataFrame()
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    sol = 0
    log = 0
    for i in range(10):
        if algorithm == "eaSimple":
            sol, log = algorithms.eaSimple(population=pop, toolbox=toolbox, cxpb=1.0, mutpb=0.5, ngen=100, stats=stats, halloffame=hof, verbose=True)
        elif algorithm == "eaMuPlusLambda":
            sol, log = algorithms.eaMuPlusLambda(population=pop, toolbox=toolbox, mu=6, lambda_=10,  cxpb=0.5, mutpb=0.5, ngen=100, stats=stats, halloffame=hof, verbose=True)
        else:
            sol, log = algorithms.eaMuCommaLambda(population=pop, toolbox=toolbox, mu=6, lambda_=10,  cxpb=0.5, mutpb=0.5, ngen=100, stats=stats, halloffame=hof, verbose=True)
        df2 = pd.DataFrame(log)
        df2['algoritmo'] = algorithm
        df2['corrida'] = i

        for i in range(1, len(df2)):
            if df2['max'][i] < df2['max'][i-1]:
                df2['max'][i] = df2['max'][i-1]
        df = df.append(df2)

    df = df.reset_index(drop=True)
    df_promedios = df.groupby(['algoritmo', 'gen']).agg({'max': ['mean', 'std']})   
    # print(df_promedios.to_string())
    print(hof) 
    x = df['gen'].unique()
    promedios = df_promedios['max']['mean'].values
    desviacion = df_promedios['max']['std'].values
    plt.plot(x, promedios, color='r')
    plt.plot(x, promedios - desviacion, linestyle='--', color='b')
    plt.plot(x, promedios + desviacion, linestyle='--', color='g')
    plt.show()
