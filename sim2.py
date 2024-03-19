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

@hydra.main(config_path=".", config_name="fisher_model_params")
def main(cfg: DictConfig):
    rng = Generator(PCG64(cfg.seed))

    def fitness(population, optimal_traits=np.array(cfg.init_opt_gen), sel_std=cfg.sel_std):
        distances = np.linalg.norm(population - optimal_traits, axis=1)
        return np.exp(-distances / (2 * sel_std ** 2))

    def mutate(population, p_mut=cfg.p_mut, mut_std=cfg.mut_std):
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

    def select_parents(population, fitnesses, comp_rate=cfg.competition_rate):
        parents = [element for element, prob in zip(population, fitnesses) if rng.uniform(0, 1) < prob]
        return parents

    def init_individual(size=cfg.n_traits, trait_min=cfg.trait_min, trait_max=cfg.trait_max):
        return rng.uniform(trait_min, trait_max, size)

    def init_population(pcls, ind_init, size=cfg.init_size):
        return pcls(ind_init() for _ in range(size))

    def global_warming(optimal_traits, warm_rate=cfg.warm_rate):
        return optimal_traits + rng.uniform(0, warm_rate, cfg.n_traits)

    def meteor_hit(trait_min=cfg.trait_min, trait_max=cfg.trait_max):
        return rng.uniform(trait_min, trait_max, cfg.n_traits)

    def interact(population_predator, fitnesses_predator, population_prey, fitnesses_prey):
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

    def simulate_evolution(config):
        toolbox_predator = create_toolbox()
        toolbox_prey = create_toolbox()

        population_predator = init_population(list, init_individual, config.init_size)
        population_prey = init_population(list, init_individual, config.init_size)
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

            population_predator = toolbox_predator.mutate(population_predator)
            population_prey = toolbox_prey.mutate(population_prey)

            fitnesses_predator = toolbox_predator.evaluate(population_predator, optimal_traits_predator)
            fitnesses_prey = toolbox_prey.evaluate(population_prey, optimal_traits_prey)

            history_predator["fitnesses"] = fitnesses_predator
            history_prey["fitnesses"] = fitnesses_prey

            population_predator, fitnesses_predator, population_prey, fitnesses_prey = interact(population_predator, fitnesses_predator, population_prey, fitnesses_prey)

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

            if len(population_predator) > config.max_size:
                fitnesses_predator = toolbox_predator.evaluate(population_predator, optimal_traits_predator)
                indcs = np.argsort(fitnesses_predator)
                population_predator = [population_predator[i] for i in indcs[:config.max_size]]
            elif len(population_predator) <= 0:
                print(f"Predator population died out at generation {gen}.")
                break

            if len(population_prey) > config.max_size:
                fitnesses_prey = toolbox_prey.evaluate(population_prey, optimal_traits_prey)
                indcs = np.argsort(fitnesses_prey)
                population_prey = [population_prey[i] for i in indcs[:config.max_size]]
            elif len(population_prey) <= 0:
                print(f"Prey population died out at generation {gen}.")
                break

        pop_history_prey.dropna(how='all', axis=0, inplace=True)
        pop_history_predator.dropna(how='all', axis=0, inplace=True)
        return pop_history_prey, pop_history_predator

    def grid_search_parallel(config, mut_probs, mut_stds, n_simulations=10):
        results_predator = []
        results_prey = []

        base_seed_seq = np.random.SeedSequence()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {}
            for mut_prob, mut_std in product(mut_probs, mut_stds):
                seed_seq = base_seed_seq.spawn(n_simulations)
                for seed in seed_seq:
                    new_config = config.copy()
                    new_config.p_mut = mut_prob
                    new_config.mut_std = mut_std
                    new_config.seed = seed
                    future = executor.submit(simulate_evolution, new_config)
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

    if cfg.grid_search.enabled:
        mut_probs = cfg.grid_search.mut_probs
        mut_stds = cfg.grid_search.mut_stds
        n_simulations = cfg.grid_search.n_simulations
        results_predator, results_prey = grid_search_parallel(cfg, mut_probs, mut_stds, n_simulations)
        # ... rest of the grid search code ...

    if cfg.plotting.enabled:
        # ... plotting code ...

    if cfg.animation.enabled:
        # ... animation code ...

if __name__ == "__main__":
    main()