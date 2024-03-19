import numpy as np
from numpy.random import Generator, PCG64
from deap import base
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import concurrent.futures
from itertools import product
from sklearn.decomposition import PCA

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
               rng.uniform(0, 1) < prob]
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
        if fitnesses_predator[i] > 2*fitnesses_prey[j]:
            # Predator eats prey
            indcs_dead_prey.append(j)
            fitnesses_predator[i] += 0.1 * fitnesses_prey[j]

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

    # Create a base seed sequence
    base_seed_seq = np.random.SeedSequence()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}
        for mut_prob, mut_std in product(mut_probs, mut_stds):
            # Generate unique seeds for each simulation
            seed_seq = base_seed_seq.spawn(n_simulations)

            for seed in seed_seq:
                new_config = config.copy()
                new_config['p_mut'] = mut_prob
                new_config['mut_std'] = mut_std
                new_config['seed'] = seed
                future = executor.submit(simulate_evolution, new_config)
                futures[future] = (mut_prob, mut_std)

        for future in concurrent.futures.as_completed(futures):
            mut_prob, mut_std = futures[future]
            predator_result, prey_result = future.result()
            results_predator.append((mut_prob, mut_std, predator_result.shape[0]))
            results_prey.append((mut_prob, mut_std, prey_result.shape[0]))

    # Calculate average survival rates
    avg_results_predator = {}
    avg_results_prey = {}
    for mut_prob, mut_std, survival_rate in results_predator:
        key = (mut_prob, mut_std)
        if key not in avg_results_predator:
            avg_results_predator[key] = []
        avg_results_predator[key].append(survival_rate)

    for mut_prob, mut_std, survival_rate in results_prey:
        key = (mut_prob, mut_std)
        if key not in avg_results_prey:
            avg_results_prey[key] = []
        avg_results_prey[key].append(survival_rate)

    final_results_predator = [(mut_prob, mut_std, np.mean(survival_rates))
                              for (mut_prob, mut_std), survival_rates in avg_results_predator.items()]
    final_results_prey = [(mut_prob, mut_std, np.mean(survival_rates))
                          for (mut_prob, mut_std), survival_rates in avg_results_prey.items()]

    return final_results_predator, final_results_prey


if __name__ == "__main__":

    _GRID, _PLOT, _ANIME = False, True, False

    if _GRID:
        mut_probs = np.linspace(0.1, 0.9, 10)
        mut_stds = np.linspace(0.1, 0.9, 10)
        results_predator, results_prey = grid_search_parallel(_CONFIG, mut_probs, mut_stds)
        results_predator_df = pd.DataFrame(results_predator, columns=["p_mut", "mut_std", "avg_survival_rate"])
        results_prey_df = pd.DataFrame(results_prey, columns=["p_mut", "mut_std", "avg_survival_rate"])
        results_predator_df['p_mut'] = results_predator_df['p_mut'].round(2)
        results_predator_df['mut_std'] = results_predator_df['mut_std'].round(2)
        results_prey_df['p_mut'] = results_prey_df['p_mut'].round(2)
        results_prey_df['mut_std'] = results_prey_df['mut_std'].round(2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        sns.heatmap(results_predator_df.pivot(index="p_mut", columns="mut_std", values="avg_survival_rate"),
                    annot=True, fmt='.2f', cmap='Blues', ax=ax1)
        ax1.set_title('Predator population')

        sns.heatmap(results_prey_df.pivot(index="p_mut", columns="mut_std", values="avg_survival_rate"),
                    annot=True, fmt='.2f', cmap='Blues', ax=ax2)
        ax2.set_title('Prey population')

        plt.tight_layout()
        plt.show()
        
    if _PLOT:
        new_config = _CONFIG.copy()
        new_config['p_mut'] = 0.1
        new_config['mut_std'] = 0.46
        new_config['n_generations'] = 10
        pop_history_prey = pd.DataFrame()

        iter_stop = 0
        while pop_history_prey.shape[0] < new_config['n_generations'] and iter_stop < 10:
            pop_history_prey, pop_history_predator = simulate_evolution(new_config)
            iter_stop += 1

        if iter_stop == 10 and pop_history_prey.shape[0] < new_config['n_generations']:
            print("Simulation did not converge.")
        else:

            pca = PCA(n_components=2)
            pop_history_prey['pca'] = pop_history_prey['population'].apply(lambda x: pca.fit_transform(np.array(x)))

            fig1, ax1 = plt.subplots(3, 2, figsize=(20, 30))
            for (i, j), k in zip(product(range(3), range(2)), range(0, new_config['n_generations'], 5)):
                X, Y = pop_history_prey['pca'][k].T
                ax1[i, j].scatter(X, Y, c=pop_history_prey['fitnesses'][k], cmap='seismic', alpha=0.5, label='Population')
                ax1[i, j].plot(pop_history_prey['optimal_traits'][k][0], pop_history_prey['optimal_traits'][k][1], 'x',
                              color='black', label='Optimal traits')
                ax1[i, j].set_title(f"Prey generation {k}")
                ax1[i, j].set_xlim(-1, 1)
                ax1[i, j].set_ylim(-1, 1)
                ax1[i, j].legend()
            plt.show()

            pop_history_predator['pca'] = pop_history_predator['population'].apply(lambda x: pca.fit_transform(np.array(x)))
            fig2, ax2 = plt.subplots(3, 2, figsize=(20, 30))
            for (i, j), k in zip(product(range(3), range(2)), range(0, new_config['n_generations'], 5)):
                X, Y = pop_history_predator['pca'][k].T
                ax2[i, j].scatter(X, Y, c=pop_history_predator['fitnesses'][k], cmap='seismic', alpha=0.5,
                                 label='Population')
                ax2[i, j].plot(pop_history_predator['optimal_traits'][k][0], pop_history_predator['optimal_traits'][k][1], 'x',
                              color='black', label='Optimal traits')
                ax2[i, j].set_title(f"Predator generation {k}")
                ax2[i, j].set_xlim(-1, 1)
                ax2[i, j].set_ylim(-1, 1)
                ax2[i, j].legend()
            plt.show()

    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from sklearn.decomposition import PCA

    if _ANIME:
        new_config = _CONFIG.copy()
        new_config['p_mut'] = 0.81
        new_config['mut_std'] = 0.81
        new_config['n_generations'] = 5000
        pop_history_prey = pd.DataFrame()

        iter_stop = 0
        while pop_history_prey.shape[0] < new_config['n_generations'] and iter_stop < 100:
            pop_history_prey, pop_history_predator = simulate_evolution(new_config)
            iter_stop += 1

        if iter_stop == 100 and pop_history_prey.shape[0] < new_config['n_generations']:
            print("Simulation did not converge.")
        else:
            pca = PCA(n_components=2)
            pop_history_prey['pca'] = pop_history_prey['population'].apply(lambda x: pca.fit_transform(np.array(x)))
            pop_history_predator['pca'] = pop_history_predator['population'].apply(lambda x: pca.fit_transform(np.array(x)))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


            def animate(i):
                ax1.clear()
                ax2.clear()

                X, Y = pop_history_prey['pca'][i].T
                ax1.scatter(X, Y, c=pop_history_prey['fitnesses'][i], cmap='seismic', alpha=0.5, label='Population')
                ax1.plot(pop_history_prey['optimal_traits'][i][0], pop_history_prey['optimal_traits'][i][1], 'x',
                         color='black', label='Optimal traits')
                ax1.set_title(f"Prey generation {i}")
                ax1.set_xlim(-1, 1)
                ax1.set_ylim(-1, 1)
                ax1.legend()

                X, Y = pop_history_predator['pca'][i].T
                ax2.scatter(X, Y, c=pop_history_predator['fitnesses'][i], cmap='seismic', alpha=0.5, label='Population')
                ax2.plot(pop_history_predator['optimal_traits'][i][0], pop_history_predator['optimal_traits'][i][1], 'x',
                         color='black', label='Optimal traits')
                ax2.set_title(f"Predator generation {i}")
                ax2.set_xlim(-1, 1)
                ax2.set_ylim(-1, 1)
                ax2.legend()


        ani = FuncAnimation(fig, animate, frames=new_config['n_generations'], interval=200)
        plt.tight_layout()
        plt.show()

        





