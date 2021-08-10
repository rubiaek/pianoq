import numpy as np


class Swarm(object):
    def __init__(self, cost_func, n_pop, n_var, w, wdamp, c1, c2, bounds=None, sample_func=None):
        self.cost_func = cost_func
        self.n_pop = n_pop
        self.n_var = n_var
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.lower_bound, self.upper_bound = bounds or (0, 1)
        self.sample_func = sample_func or rand_sample
        self.vel_max = 0.1 * (self.upper_bound - self.lower_bound)
        self.vel_min = -self.vel_max

        self.particles = []
        self.global_besl_positions = (self.upper_bound - self.lower_bound) * np.ones(self.n_var)
        self.global_besl_cost = cost_func(self.global_besl_positions)

    def populate_particles(self):
        for i in range(self.n_pop):
            p = Particle(self)
            p.positions = self.sample_func(size=self.n_var)
            p.velocities = np.zeros(self.dim)
            p.best_positions = p.positions.copy()
            cost = self.cost_function(p.positions)
            p.cost = cost
            p.best_cost = cost
            self.particles.append(p)

            if p.best_cost < self.global_besl_cost:
                self.global_besl_cost = p.best_cost
                self.global_besl_positions = p.positions


    def reduce_population(self):
        pass

    def do_iteration(self):
        # At end of iteration we will have a new global_best_positions and global_best_cost
        for particle in self.particles:
            particle.try_again()


class Particle(object):
    def __init__(self, swarm):
        self.swarm = swarm
        self.dim = self.swarm.n_var
        self.positions = None
        self.velocities = None
        self.cost = None
        self.best_positions = None
        self.best_cost = None

    def try_again(self):
        pass


class MyPSOOptimizer(object):
    """ We try to MINIMIZE the cost function """
    def __init__(self, cost_function, n_pop, n_var, n_iterations, post_iteration_callback,
                 w=1, wdamp=0.99, c1=1.5, c2=2,
                 timeout=np.inf, early_stop=True, stop_after_n_const_iter=8, vary_popuation=True, slope=-1):

        self.cost_function = cost_function
        self.n_iterations = n_iterations
        # TODO: this should be a function that gets a good picture by mean on 30 pictures, and saves to the result object
        self.post_iteration_callback = post_iteration_callback

        self.swarm = Swarm(cost_func=cost_function, n_pop=n_pop, n_var=n_var,
                           w=w, wdamp=wdamp, c1=c1, c2=c2)

    def optimize(self, verbose=True):

        # Do it first at the beginning with the initial guess, then again after initial population
        self.post_iteration_callback(self.swarm.global_besl_cost, self.swarm.global_besl_positions)
        self.swarm.populate_particles()
        self.post_iteration_callback(self.swarm.global_besl_cost, self.swarm.global_besl_positions)

        for i in range(self.n_iterations):
            self.swarm.do_iteration()
            self.post_iteration_callback(self.swarm.global_besl_cost, self.swarm.global_besl_positions)
            # TODO: something with stop_after_num_iter, vary_popuation, timeout


def rand_sample(size):
    return np.random.uniform(size=size)


# TODO: I got to line 112 - PSO main loop
