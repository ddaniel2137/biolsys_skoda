import hydra
from omegaconf import DictConfig
import numpy as np
from numpy.random import Generator, PCG64
from deap import base
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
from itertools import product
from sklearn.decomposition import PCA
import seaborn as sns

def grid_search_parallel(config, mut_probs, mut_stds, n_simulations=10, rng=Generator(PCG64(2137))):
    results_predator = []
    results_prey = []
    seed_seq = range(2137, 2137 + n_simulations)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}
        for mut_prob, mut_std in product(mut_probs, mut_stds):

            for seed in seed_seq:
                new_config = config.copy()
                new_config.simulation.p_mut = mut_prob
                new_config.simulation.mut_std = mut_std
                new_config.simulation.seed = seed
                future = executor.submit(simulate_evolution, new_config.simulation, rng)
                futures[future] = (mut_prob, mut_std)

            for future in concurrent.futures.as_completed(futures):
                mut_prob, mut_std = futures[future]
                predator_result, prey_result = future.result()
                results_predator.append((mut_prob, mut_std, predator_result.shape[0]))
                results_prey.append((mut_prob, mut_std, prey_result.shape[0]))

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

def init_individual(size, trait_min, trait_max, rng):
    return rng.uniform(trait_min, trait_max, size)
    
def simulate_evolution(config, rng):

    rng = Generator(PCG64(config.seed))
    toolbox_predator = create_toolbox()
    toolbox_prey = create_toolbox()

    population_predator = init_population(list, lambda: init_individual(config.n_traits, config.trait_min, config.trait_max, rng), config.init_size)
    population_prey = init_population(list, lambda: init_individual(config.n_traits, config.trait_min, config.trait_max, rng), config.init_size)
    optimal_traits_predator = np.array(config.init_opt_gen)
    optimal_traits_prey = np.array(config.init_opt_gen)

    stat_categories = ["population", "fitnesses", "optimal_traits"]
    pop_history_predator = pd.DataFrame(columns=stat_categories, index=np.arange(config.n_generations))
    pop_history_prey = pd.DataFrame(columns=stat_categories, index=np.arange(config.n_generations))
    history_predator = dict(stat_categories=[])
    history_prey = dict(stat_categories=[])

    for gen in range(config.n_generations):
        history_predator["population"] = population_predator
        history_prey["population"] = population_prey
        history_predator["optimal_traits"] = optimal_traits_predator
        history_prey["optimal_traits"] = optimal_traits_prey
    
        population_predator = toolbox_predator.mutate(population_predator, config.p_mut, config.mut_std, rng)
        population_prey = toolbox_prey.mutate(population_prey, config.p_mut, config.mut_std, rng)

        fitnesses_predator = toolbox_predator.evaluate(population_predator, optimal_traits_predator, config.sel_std)
        fitnesses_prey = toolbox_prey.evaluate(population_prey, optimal_traits_prey, config.sel_std)

        history_predator["fitnesses"] = fitnesses_predator
        history_prey["fitnesses"] = fitnesses_prey

        population_predator, fitnesses_predator, population_prey, fitnesses_prey = interact(population_predator, fitnesses_predator, population_prey, fitnesses_prey, rng)

        parents_predator = toolbox_predator.select(population_predator, fitnesses_predator, rng)
        parents_prey = toolbox_prey.select(population_prey, fitnesses_prey, rng)

        offspring_predator = toolbox_predator.mate(parents_predator, rng)
        offspring_prey = toolbox_prey.mate(parents_prey, rng)

        population_predator = parents_predator + offspring_predator
        population_prey = parents_prey + offspring_prey

        optimal_traits_predator = global_warming(optimal_traits_predator, config.warm_rate, rng)
        optimal_traits_prey = global_warming(optimal_traits_prey, config.warm_rate, rng)

        if rng.uniform(0, 1) < 0.005:
            optimal_traits_predator, optimal_traits_prey = meteor_hit(config.trait_min, config.trait_max, config.n_traits, rng), meteor_hit(config.trait_min, config.trait_max, config.n_traits, rng)
            print(f"Meteor hit at generation {gen}.")

        pop_history_predator.loc[gen] = history_predator.copy()
        pop_history_prey.loc[gen] = history_prey.copy()

        if len(population_predator) > config.max_size:
            fitnesses_predator = toolbox_predator.evaluate(population_predator, optimal_traits_predator, config.sel_std)
            indcs = np.argsort(fitnesses_predator)
            population_predator = [population_predator[i] for i in indcs[:config.max_size]]
        elif len(population_predator) <= 0:
            print(f"Predator population died out at generation {gen}.")
            break

        if len(population_prey) > config.max_size:
            fitnesses_prey = toolbox_prey.evaluate(population_prey, optimal_traits_prey, config.sel_std)
            indcs = np.argsort(fitnesses_prey)
            population_prey = [population_prey[i] for i in indcs[:config.max_size]]
        elif len(population_prey) <= 0:
            print(f"Prey population died out at generation {gen}.")
            break

    pop_history_prey.dropna(how='all', axis=0, inplace=True)
    pop_history_predator.dropna(how='all', axis=0, inplace=True)
    return pop_history_prey, pop_history_predator



def fitness(population, optimal_traits, sel_std):
    distances = np.linalg.norm(population - optimal_traits, axis=1)
    return np.exp(-distances / (2 * sel_std ** 2))

def mutate(population, p_mut, mut_std, rng):
    for i, individual in enumerate(population):
        if rng.uniform(0, 1) < p_mut:
            ind = rng.integers(0, individual.size)
            mut_val = rng.normal(0, mut_std ** 2)
            individual[ind] += mut_val
    return population

def reproduce(parents, rng):
    n = len(parents)
    n_half = n // 2
    rng.shuffle(parents)
    group1, group2 = parents[:n_half], parents[n_half:]
    offspring = []
    for p1, p2 in zip(group1, group2):
        child = (p1 + p2) / rng.uniform(2, 4)
        offspring.append(child)
    return offspring

def select_parents(population, fitnesses, rng):
    parents = [element for element, prob in zip(population, fitnesses) if rng.uniform(0, 1) < prob]
    return parents

def init_population(pcls, ind_init, size):
    return pcls(ind_init() for _ in range(size))

def global_warming(optimal_traits, warm_rate, rng):
    return optimal_traits + rng.uniform(0, warm_rate, optimal_traits.size)

def meteor_hit(trait_min, trait_max, n_traits, rng):
    return rng.uniform(trait_min, trait_max, n_traits)

def interact(population_predator, fitnesses_predator, population_prey, fitnesses_prey, rng):
    indcs_predator = rng.choice(len(population_predator), min(len(population_predator) // 2, len(population_prey) // 2), replace=False)
    indcs_prey = rng.choice(len(population_prey), min(len(population_predator) // 2, len(population_prey) // 2), replace=False)
    indcs_dead_prey = []
    for i, j in zip(indcs_predator, indcs_prey):
        if fitnesses_predator[i] > 2*fitnesses_prey[j]:
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


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig):

    rng = Generator(PCG64(cfg.simulation.seed))

    def plot_evolution(config, pop_history_prey, pop_history_predator):
        pca = PCA(n_components=2)

        def plot_population(pop_history, ax, title):
            generations = pop_history.index
            for i, gen in enumerate(generations[:min(len(ax), len(generations))]):
                population = pop_history.loc[gen, 'population']
                fitnesses = pop_history.loc[gen, 'fitnesses']
                optimal_traits = pop_history.loc[gen, 'optimal_traits']

                if len(population) >= 2:
                    pca_transformed = pca.fit_transform(np.array(population))
                    X, Y = pca_transformed.T

                    population_scatter = ax[i // 2, i % 2].scatter(X, Y, c=fitnesses, cmap='seismic', alpha=0.5, label='Population')
                    optimal_traits_plot = ax[i // 2, i % 2].plot(optimal_traits[0], optimal_traits[1], 'x', color='black', label='Optimal traits')
                else:
                    ax[i // 2, i % 2].text(0.5, 0.5, 'Insufficient data', horizontalalignment='center', verticalalignment='center')

                ax[i // 2, i % 2].set_title(f"{title} generation {gen}")
                ax[i // 2, i % 2].set_xlim(-1, 1)
                ax[i // 2, i % 2].set_ylim(-1, 1)

                if i == 0:
                    ax[i // 2, i % 2].legend()

            return ax

        if pop_history_prey.empty or pop_history_predator.empty:
            print("Population history is empty. Cannot plot evolution.")
            return

        fig1, ax1 = plt.subplots(3, 2, figsize=(20, 30))
        ax1 = plot_population(pop_history_prey, ax1, "Prey")
        plt.tight_layout()
        plt.show()

        fig2, ax2 = plt.subplots(3, 2, figsize=(20, 30))
        ax2 = plot_population(pop_history_predator, ax2, "Predator")
        plt.tight_layout()
        plt.show()

    if cfg.grid_search.enabled:
        mut_probs = cfg.grid_search.mut_probs
        mut_stds = cfg.grid_search.mut_stds
        n_simulations = cfg.grid_search.n_simulations
        results_predator, results_prey = grid_search_parallel(cfg, mut_probs, mut_stds, n_simulations, rng)
        results_predator_df = pd.DataFrame(results_predator, columns=["mut_prob", "mut_std", "avg_survival_rate"])
        results_prey_df = pd.DataFrame(results_prey, columns=["mut_prob", "mut_std", "avg_survival_rate"])
        results_predator_df['mut_prob'] = results_predator_df['mut_prob'].round(2)
        results_predator_df['mut_std'] = results_predator_df['mut_std'].round(2)
        results_prey_df['mut_prob'] = results_prey_df['mut_prob'].round(2)
        results_prey_df['mut_std'] = results_prey_df['mut_std'].round(2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        sns.heatmap(results_predator_df.pivot(index="mut_prob", columns="mut_std", values="avg_survival_rate"),
                    annot=True, fmt='.2f', cmap='Blues', ax=ax1)
        ax1.set_title('Predator population')

        sns.heatmap(results_prey_df.pivot(index="mut_prob", columns="mut_std", values="avg_survival_rate"),
                    annot=True, fmt='.2f', cmap='Blues', ax=ax2)
        ax2.set_title('Prey population')

        plt.tight_layout()
        plt.show()

    if cfg.plotting.enabled:
        pop_history_prey, pop_history_predator = simulate_evolution(cfg.simulation, rng)
        plot_evolution(cfg, pop_history_prey, pop_history_predator)

    if cfg.animation.enabled:
        pass

if __name__ == "__main__":
    main()