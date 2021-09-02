from functools import reduce

from pianoq.lab.optimizations.my_pso import MyPSOOptimizer
from pianoq.results import PopoffPRXResult


class PianoPopoffSimulation(object):
    def __init__(self, piezo_num=30):
        self.piezo_num = piezo_num

        self.pop = PopoffPRXResult(path=PopoffPRXResult.DEFAULT_PATH)

        # We will let the PSO algorithm live in a continuous [0,1] world, and translate it to one of the discreet set
        # of dxs, which will make us choose the relevant TM
        self.dxs = self.pop.TM_modes[self.pop.index_dx0:]
        self.TMs = self.pop.TM_modes[self.pop.index_dx0:]

        self.optimizer = None
        self.n_pop = None


    def run(self, n_pop=30, cost_function=None):
        cost_function = cost_function or self.cost_function
        self.n_pop = n_pop
        self.optimizer = MyPSOOptimizer(cost_function, n_pop=n_pop, n_var=self.num_good_piezos,
                                        n_iterations=n_iterations,
                                        post_iteration_callback=self.post_iteration_callback,
                                        w=1, wdamp=0.99, c1=1.5, c2=2,
                                        timeout=30*60,
                                        stop_early=True, stop_after_n_const_iter=stop_after_n_const_iters,
                                        vary_popuation=True, reduce_at_iterations=reduce_at_iterations)

    def cost_function_focus(self, amps):
        """ amps are between 0 and 1 """


        # (1) initialize some input beam in the mode basis
        in_modes = np.ones(self.pop.Nmodes)

        # (2) translate the amps to dicreet dx values
        # dx values are between 0 to 70 with jumps of 2: [0, 2, 4, ..., 68, 70]
        amps = amps * 35 # now between [0, 35]
        TM_indexes = np.around(amps).astype(int)

        # (3) propagate the input beam through the relevant TMs (according to (2) dxs)
        curr_TMs = [self.TMs[i] for i in TM_indexes]
        tot_curr_TM = reduce(lambda x, y: x @ y, curr_TMs)


        # (4) translate the output in mode basis to pixel basis
        pix1, pix2 = self.pop.propagate(in_modes, tot_curr_TM)

        # (5) "measure" the intensity in some point


        pass
