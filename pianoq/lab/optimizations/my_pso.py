import numpy as np


class Swarm(object):
    def __init__(self, n_pop, n_var, w, c1, c2, bounds=None, sample_func=None):
        self.n_pop = n_pop
        self.n_var = n_var
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.lower_bound, self.upper_bound = bounds or (0, 1)
        self.sample_func = sample_func

        self.particles = []
        for i in range(n_pop):
            p = Particle(self.n_var)
            self.particles.append(p)

    def init_particles(self):
        pass

    def do_iteration(self):
        pass


class Particle(object):
    def __init__(self, dim):
        self.dim = dim
        self.positions = None
        self.velocities = None
        self.best_positoins = None
        self.best_velocities = None


def optimize(piano_opt, n_pop, iterations, w, c1, c2):
    pass
