import numpy as np
from numpy.random import Generator, PCG64
from deap import base
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import concurrent.futures
from itertools import product

_CONFIG = {
    'init_size': 100,
    'max_size': 1000,
    'n_generations': 50,
    'n_traits': 3,
    'trait_min': -1,
    'trait_max': 1,
    'p_mut': 0.1,
    'mut_std': 0.1,
    'init_opt_gen': np.zeros(3),
    'sel_std': 0.95,
    'warm_rate': 0.01,
    'competition_rate': 0.1,
    'seed': 2137
}

rng = Generator(PCG64(_CONFIG['seed']))


def fitness(population, optimal_traits=_CONFIG['init_opt_gen'], sel_std=_CONFIG['sel_std']):
    distances = np.linalg.norm(population - optimal_traits, axis=1)
    return np.exp(-distances / (2 * sel_std ** 2))


def mutate(population, p_mut=_CONFIG['p_mut'], mut_std=_CONFIG['mut_std']):
    for i, individual in enumerate(population):
        if rng.uniform(0, 1) < p_mut:
            ind = rng.integers(0, individual.size)
            mut_val = rng.normal(0, mut_std ** 2)
            individual[ind] += mut_val

    return population


def reproduce(parents):
    n = len(parents)
    n_half = n // 2
    rng.shuffle(parents)
    group1, group2 = parents[:n_half], parents[n_half:]
    offspring = []
    for p1, p2 in zip(group1, group2):
        child = (p1 + p2) / rng.uniform(2, 4)
        offspring.append(child)
    return offspring


def select_parents(population, fitnesses, comp_rate=_CONFIG['competition_rate']):
    parents = [element for element, prob in zip(population, fitnesses) if
               rng.uniform(0, 1) < (1 - _CONFIG['competition_rate']) * prob]
    return parents


def init_individual(size=_CONFIG['n_traits'], trait_min=_CONFIG['trait_min'], trait_max=_CONFIG['trait_max']):
    return rng.uniform(trait_min, trait_max, size)


def init_population(pcls, ind_init, size=_CONFIG['init_size']):
    return pcls(ind_init() for _ in range(size))


def global_warming(optimal_traits, warm_rate=_CONFIG['warm_rate']):
    return optimal_traits + rng.uniform(0, warm_rate, _CONFIG['n_traits'])


def meteor_hit(trait_min=_CONFIG['trait_min'], trait_max=_CONFIG['trait_max']):
    return rng.uniform(trait_min, trait_max, _CONFIG['n_traits'])


def interact(population_predator, fitnesses_predator, population_prey, fitnesses_prey):
    # Define interaction between preys and predators
    indcs_predator = rng.choice(len(population_predator), min(len(population_predator) // 2, len(population_prey) // 2),
                                                              replace=False)
    indcs_prey = rng.choice(len(population_prey), min(len(population_predator) // 2, len(population_prey) // 2),
                                                              replace=False)
    #print(f"Predator indices: {indcs_predator}, Prey indices: {indcs_prey}")
    indcs_dead_prey = []
    for i, j in zip(indcs_predator, indcs_prey):
        if fitnesses_predator[i] > fitnesses_prey[j]:
            # Predator eats prey
            indcs_dead_prey.append(j)
            fitnesses_predator[i] += 0.5 * fitnesses_prey[j]

    population_prey = [population_prey[i] for i in range(len(population_prey)) if i not in indcs_dead_prey]
    indcs_dead_prey = np.array(indcs_dead_prey, dtype=int)
    mask = np.ones(len(fitnesses_prey), dtype=bool)
    mask[indcs_dead_prey] = False
    fitnesses_prey = fitnesses_prey[mask]
    return population_predator, fitnesses_predator, population_prey, fitnesses_prey


def create_toolbox():
    toolbox = base.Toolbox()
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", reproduce)
    toolbox.register("mutate", mutate)
    toolbox.register("select", select_parents)
    return toolbox


def simulate_evolution(config):
    toolbox_predator = create_toolbox()
    toolbox_prey = create_toolbox()

    population_predator = init_population(list, init_individual, config['init_size'])
    population_prey = init_population(list, init_individual, config['init_size'])
    optimal_traits_predator = config['init_opt_gen']
    optimal_traits_prey = config['init_opt_gen']

    stat_categories = ["population", "fitnesses", "optimal_traits"]
    pop_history_predator = pd.DataFrame(columns=stat_categories, index=np.arange(config['n_generations']))
    pop_history_prey = pd.DataFrame(columns=stat_categories, index=np.arange(config['n_generations']))
    history_predator = dict(stat_categories=[])
    history_prey = dict(stat_categories=[])

    for gen in range(config['n_generations']):

        history_predator["population"] = population_predator
        history_prey["population"] = population_prey
        history_predator["optimal_traits"] = optimal_traits_predator
        history_prey["optimal_traits"] = optimal_traits_prey

        population_predator = toolbox_predator.mutate(population_predator)
        population_prey = toolbox_prey.mutate(population_prey)

        fitnesses_predator = toolbox_predator.evaluate(population_predator, optimal_traits_predator)
        fitnesses_prey = toolbox_prey.evaluate(population_prey, optimal_traits_prey)

        history_predator["fitnesses"] = fitnesses_predator
        history_prey["fitnesses"] = fitnesses_prey
        
        population_predator, fitnesses_predator, population_prey, fitnesses_prey = interact(population_predator,
                                                                                            fitnesses_predator,
                                                                                            population_prey,
                                                                                            fitnesses_prey)
        parents_predator = toolbox_predator.select(population_predator, fitnesses_predator)
        parents_prey = toolbox_prey.select(population_prey, fitnesses_prey)

        offspring_predator = toolbox_predator.mate(parents_predator)
        offspring_prey = toolbox_prey.mate(parents_prey)

        population_predator = parents_predator + offspring_predator
        population_prey = parents_prey + offspring_prey

        optimal_traits_predator = global_warming(optimal_traits_predator)
        optimal_traits_prey = global_warming(optimal_traits_prey)

        if rng.uniform(0, 1) < 0.005:
            optimal_traits_predator, optimal_traits_prey = meteor_hit(), meteor_hit()
            print(f"Meteor hit at generation {gen}.")

        pop_history_predator.loc[gen] = history_predator.copy()
        pop_history_prey.loc[gen] = history_prey.copy()

        if len(population_predator) > config['max_size']:
            fitnesses_predator = toolbox_predator.evaluate(population_predator, optimal_traits_predator)

            indcs = np.argsort(fitnesses_predator)
            population_predator = [population_predator[i] for i in indcs[:config['max_size']]]

        elif len(population_predator) <= 0:
            print(f"Predator population died out at generation {gen}.")
            break

        if len(population_prey) > config['max_size']:
            fitnesses_prey = toolbox_prey.evaluate(population_prey, optimal_traits_prey)

            indcs = np.argsort(fitnesses_prey)
            population_prey = [population_prey[i] for i in indcs[:config['max_size']]]

        elif len(population_prey) <= 0:
            print(f"Prey population died out at generation {gen}.")
            break

    # pop_history.dropna(how='all', axis=0, inplace=True)
    pop_history_prey.dropna(how='all', axis=0, inplace=True)
    pop_history_predator.dropna(how='all', axis=0, inplace=True)
    return pop_history_prey, pop_history_predator

def grid_search_parallel(config, mut_probs, mut_stds, n_simulations=10):
    results_predator = []
    results_prey = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []

        for mut_prob, mut_std in product(mut_probs, mut_stds):
            new_config = config.copy()
            new_config['p_mut'] = mut_prob
            new_config['mut_std'] = mut_std
            future = executor.submit(simulate_evolution, new_config)
            futures.append((mut_prob, mut_std, future))
            #print(future.result())
    
        for mut_prob, mut_std, future in futures:
            pop_history_predator, pop_history_prey = future.result()
            avg_survival_rate_predator = pop_history_predator.shape[0] / n_simulations
            avg_survival_rate_prey = pop_history_prey.shape[0] / n_simulations
            results_predator.append((mut_prob, mut_std, avg_survival_rate_predator))
            results_prey.append((mut_prob, mut_std, avg_survival_rate_prey))

    return results_predator, results_prey

if __name__ == "__main__":
    mut_probs = np.linspace(0.1, 0.9, 10)
    mut_stds = np.linspace(0.1, 0.9, 10)
    results_predator, results_prey = grid_search_parallel(_CONFIG, mut_probs, mut_stds)
    results_predator_df = pd.DataFrame(results_predator, columns=["p_mut", "mut_std", "avg_survival_rate"])
    results_prey_df = pd.DataFrame(results_prey, columns=["p_mut", "mut_std", "avg_survival_rate"])
    results_predator_df['p_mut'] = results_predator_df['p_mut'].round(2)
    results_predator_df['mut_std'] = results_predator_df['mut_std'].round(2)
    results_prey_df['p_mut'] = results_prey_df['p_mut'].round(2)
    results_prey_df['mut_std'] = results_prey_df['mut_std'].round(2)

    sns.heatmap(results_predator_df.pivot(index="p_mut", columns="mut_std", values="avg_survival_rate"), annot=True, fmt='.2f', cmap='Blues')
    plt.show()
    sns.heatmap(results_prey_df.pivot(index="p_mut", columns="mut_std", values="avg_survival_rate"), annot=True, fmt='.2f', cmap='Blues')
    plt.show()
