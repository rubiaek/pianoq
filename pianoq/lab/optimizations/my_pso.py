import numpy as np
import time


class Swarm(object):
    def __init__(self, cost_func, n_pop, n_var, w, wdamp, c1, c2, bounds=None, sample_func=None):
        self.cost_func = cost_func
        self.n_pop = n_pop
        self.n_var = n_var
        self.w = w
        self.wdamp = wdamp
        self.c1 = c1
        self.c2 = c2
        self.lower_bound, self.upper_bound = bounds or (0, 1)
        self.sample_func = sample_func or rand_sample
        self.vel_max = 0.1 * (self.upper_bound - self.lower_bound)
        self.vel_min = -self.vel_max

        self.particles = []
        self.global_best_positions = (self.upper_bound - self.lower_bound) * np.ones(self.n_var)
        self.global_best_cost = cost_func(self.global_best_positions)
        self.best_particle = None

        self.iterations_since_restart_occured = 0

    def populate_particles(self):
        for i in range(self.n_pop):
            p = Particle(self)
            p.restart()
            self.particles.append(p)

            if p.best_cost < self.global_best_cost:
                self.global_best_cost = p.best_cost
                self.global_best_positions = p.positions

        self.best_particle = self.particles[0]

    def reduce_population(self, reduction_factor):
        print(f"population of {len(self.particles)}")
        sorted_particles = sorted(self.particles, key=lambda x: x.cost)
        new_len = len(self.particles) // reduction_factor
        self.particles = sorted_particles[:new_len]
        self.n_pop = new_len
        print(f"population of {len(self.particles)} after reduction")

    def update_best_particle(self, particle):
        self.best_particle = particle

    def do_iteration(self):
        self.w = self.w * self.wdamp

        restart_occurred = False
        particles_to_mutate = np.random.choice(self.n_pop, 2)  # TODO: 2 to var
        # At end of iteration we will have a new global_best_positions and global_best_cost
        for i, particle in enumerate(self.particles):
            if self.is_stuck() and particle is not self.best_particle:
                particle.restart()
                restart_occurred = True

            elif i in particles_to_mutate and particle is not self.best_particle:
                particle.mutate()
            else:
                particle.evolve()
            particle.evaluate()

        self.iterations_since_restart_occured += 1
        if restart_occurred:
            self.iterations_since_restart_occured = 0

    def is_stuck(self):
        a_condition = self.iterations_since_restart_occured > 15
        b_condition = self.particles_are_clustered()
        return a_condition and b_condition

    def particles_are_clustered(self):
        pop_mat = np.zeros((self.n_pop, self.n_var))  # Every row is a particle
        for i, particle in enumerate(self.particles):
            pop_mat[i, :] = particle.positions
        std_for_different_piezos = pop_mat.std(axis=0)
        mean_std = std_for_different_piezos.mean()
        return mean_std > 0.03


class Particle(object):
    def __init__(self, swarm):
        self.swarm = swarm
        self.dim = self.swarm.n_var
        self.positions = None
        self.velocities = None
        self.cost = None
        self.best_positions = None
        self.best_cost = None

    def evolve(self):
        # Calc velocities
        # Normalized diff from global best
        diff = (self.swarm.global_best_cost - self.cost) / (self.swarm.global_best_cost - 0.003)

        curr_vel_contrib = self.swarm.w * self.velocities
        to_self_best_contrib = self.swarm.c1 * rand_sample(size=self.dim) * (self.best_positions - self.positions)
        to_glob_best_contrib = self.swarm.c2 * diff * (self.swarm.global_best_positions - self.positions)

        self.velocities = curr_vel_contrib + to_self_best_contrib + to_glob_best_contrib

        self.velocities = np.clip(self.velocities, self.swarm.vel_min, self.swarm.vel_max)

        # Change positions
        self.positions = self.positions + self.velocities

        are_outside_indexes = np.logical_or(self.positions > self.swarm.upper_bound,
                                            self.positions < self.swarm.lower_bound)

        # Velocity Mirror effect
        self.velocities[are_outside_indexes] = -self.velocities[are_outside_indexes]

        self.positions = np.clip(self.positions, self.swarm.lower_bound, self.swarm.upper_bound)

    def is_stuck(self):
        # TODO: line 138, understand "stuck" condition and what to do in that case...
        # I think stuck in this context means that the positions didn't move a lot lately
        # (Or maybe cost didn't change a lot lately?)
        return  False

    def mutate(self):
        n_piezos_to_mutate = 3  # TODO: var to self.var
        piezos_to_mutate = np.random.choice(self.dim, n_piezos_to_mutate)
        new_positions = rand_sample(size=n_piezos_to_mutate)
        self.positions[piezos_to_mutate] = new_positions

    def restart(self):
        self.positions = self.swarm.sample_func(size=self.dim)
        self.velocities = np.zeros(self.dim)
        self.best_positions = self.positions.copy()
        cost = self.swarm.cost_func(self.positions)
        self.cost = cost
        self.best_cost = cost

    def evaluate(self):
        cost = self.swarm.cost_func(self.positions)
        self.cost = cost
        if self.cost < self.best_cost:
            self.best_cost = self.cost
            self.best_positions = self.positions

        if self.cost < self.swarm.global_best_cost:
            self.swarm.global_best_cost = self.best_cost
            self.swarm.global_best_positions = self.best_positions
            self.swarm.update_best_particle(self)


class MyPSOOptimizer(object):
    """ We try to MINIMIZE the cost function """
    def __init__(self, cost_function, n_pop, n_var, n_iterations, post_iteration_callback=None,
                 w=1, wdamp=0.99, c1=1.5, c2=2,
                 timeout=np.inf, stop_early=True, stop_after_n_const_iter=8, vary_popuation=True,
                 reduce_at_generations=None, slope=-1):

        self.cost_function = cost_function
        self.n_iterations = n_iterations

        self.stop_early = stop_early
        self.stop_after_n_const_iter = stop_after_n_const_iter

        self.vary_popuation = vary_popuation
        self.reduce_at_generations = reduce_at_generations or [4, 7]
        if vary_popuation:
            n_pop = n_pop * 2**len(reduce_at_generations)

        self.timeout = timeout

        self.curr_iteration = 0
        self.start_time = time.time()

        self.post_iteration_callback = post_iteration_callback or self.default_post_iteration

        self.swarm = Swarm(cost_func=cost_function, n_pop=n_pop, n_var=n_var,
                           w=w, wdamp=wdamp, c1=c1, c2=c2)

    def optimize(self):

        # Do it first at the beginning with the initial guess, then again after initial population
        self.post_iteration_callback(self.swarm.global_best_cost, self.swarm.global_best_positions)
        self.curr_iteration += 1
        self.swarm.populate_particles()
        self.post_iteration_callback(self.swarm.global_best_cost, self.swarm.global_best_positions)

        n_const_iter = 0
        curr_global_best = 0

        for i in range(self.n_iterations):
            self.curr_iteration += 1
            self.swarm.do_iteration()
            self.post_iteration_callback(self.swarm.global_best_cost, self.swarm.global_best_positions)

            if self.vary_popuation and self.curr_iteration in self.reduce_at_generations:
                self.swarm.reduce_population(reduction_factor=2)

            if self.stop_early:
                if self.swarm.global_best_cost == curr_global_best:
                    n_const_iter += 1
                else:
                    n_const_iter = 0
                    curr_global_best = self.swarm.global_best_cost

                if n_const_iter >= self.stop_after_n_const_iter:
                    break

            if time.time() - self.start_time > self.timeout:
                print("## TIMED OUT! ##")
                break

    def default_post_iteration(self, global_best_cost, global_best_positions):
        print(f'{self.curr_iteration}.\t cost: {global_best_cost}\t time: {(time.time()-self.start_time):2f} seconds')


def rand_sample(size):
    return np.random.uniform(size=size)

