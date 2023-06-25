import numpy as np
import matplotlib.pyplot as plt
import datetime

from pianoq.simulations import PianoPopoffSimulation


class TMShapingSimulation(object):
    def __init__(self):
        self.piezos_to_try = (10, 30, 100, 400)
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.normalize_TMs_method = 'svd1'
        self.pso_n_pop = 40
        self.N_bends = 'fiber2'

        self.saveto_path = None

    def run(self, n_mean=5):
        for piezo_num in self.piezos_to_try:
            cost_mean, cost_std, sample_before, sample_after = self.get_cost(n_mean, piezo_num)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(f'piezo_num: {piezo_num}, ratio: {cost_mean:.3f} +- {cost_std:.3f}')
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


    def get_cost(self, n_mean, piezo_num):
        costs = np.zeros(n_mean)
        sample_before, sample_after = None, None

        for i in range(n_mean):
            print(f'##### {i} ####')
            piano_sim = PianoPopoffSimulation(piezo_num=piezo_num, N_bends=self.N_bends,
                                              normalize_cost_to_tot_power=True, prop_random_phases=True,
                                              normalize_TMs_method=self.normalize_TMs_method,
                                              quiet=True)

            cost_function = piano_sim.cost_function_TM_shaping

            piano_sim.run(n_pop=self.pso_n_pop, n_iterations=1000, cost_function=cost_function,
                          stop_after_n_const_iters=10)

            amps = piano_sim.optimizer.swarm.global_best_positions
            cost = piano_sim.optimizer.swarm.global_best_cost

            print(f'cost_func value with {piezo_num} piezos: {cost}')
            costs[i] = -cost

        return costs.mean(), costs.std(), sample_before, sample_after


if __name__ == "__main__":
    n = TMShapingSimulation()
    n.run(n_mean=3)

    plt.show()
