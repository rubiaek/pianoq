"""
        elif algo == "genetic_algorithm":
            from deap import base, creator, tools, algorithms
            import random
            import numpy as np

            # Check if FitnessMax and Individual classes already exist to avoid redefinition warnings
            if not hasattr(creator, "FitnessMax"):
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            if not hasattr(creator, "Individual"):
                creator.create("Individual", list, fitness=creator.FitnessMax)

            # Initialize DEAP toolbox
            toolbox = base.Toolbox()
            toolbox.register("attr_phase", random.uniform, 0, 2 * np.pi)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_phase, n=self.N)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", lambda ind: (cost_func(ind),))  # Fitness function
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Genetic algorithm parameters
            population = toolbox.population(n=100)
            max_generations = 2000  # Set this to your desired number of generations
            algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=max_generations, verbose=False)

            best_individual = tools.selBest(population, k=1)[0]

            # Update slm_phases with best individual found
            self.slm_phases = np.array(best_individual)
            intensity = cost_func(self.slm_phases)
            return intensity, population

        elif algo == "PSO":
            from pianoq.lab.optimizations.my_pso import MyPSOOptimizer

            def cost_func2(phases):
                return cost_func(phases), None, None

            o = MyPSOOptimizer(cost_func2, n_pop=80, n_var=self.N, n_iterations=1000, stop_early=True,
                               stop_after_n_const_iter=100, quiet=True)
            o.optimize()
            self.slm_phases = o.best_positions
            intensity = cost_func(self.slm_phases)
            return intensity, o

"""