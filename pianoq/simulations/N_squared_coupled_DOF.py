import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from pianoq.lab.optimizations.my_pso import MyPSOOptimizer


class NsqrDOFSimulation():
    def __init__(self, N, M=None):
        self.N = N
        self.M = M if M else N
        self.U_random = unitary_group.rvs(self.N)
        self.U_desired = unitary_group.rvs(self.N)


    def compare_unitaries(self, U1, U2):
        # https://math.stackexchange.com/questions/1958164/measure-to-compare-a-matrix-with-a-given-unitary-matrix
        nominator = np.trace(U1.conjugate().T@U2)
        N = U1.shape[0]
        denominator = np.sqrt(N*np.trace(U2.conjugate().T@U2))
        X = nominator / denominator
        return np.abs(X)**2

    def amps_to_mat(self, V):
        # V is vector with N^2 amplitudes between 0 and 1
        V = V * 2 * np.pi
        M = np.reshape(V, (self.N, self.M))

        final = np.eye(self.N)
        for i in range(self.M):
            final = final@self.U_random
            final = final@np.diag(np.exp(1j*M[:, i]))
        return final

    def cost_function(self, V):
        # V is vector with N*M amplitudes between 0 and 1
        mat = self.amps_to_mat(V)
        cost = self.compare_unitaries(mat, self.U_desired)
        return -cost, None, None

    def show_results(self):
        best_amps = self.optimizer.best_positions
        mat_best = self.amps_to_mat(best_amps)
        best_cost = np.abs(self.optimizer.best_cost)

        print("########")
        print(self.compare_unitaries(mat_best, self.U_desired))
        print("########")
        fig, axes = plt.subplots(4, 2)
        axes[0, 0].imshow(mat_best.real)
        axes[0, 0].set_title('real best optimized')
        axes[0, 1].imshow(self.U_desired.real)
        axes[0, 1].set_title('real desired random target')
        axes[1, 0].imshow(mat_best.imag)
        axes[1, 0].set_title('imag best optimized')
        axes[1, 1].imshow(self.U_desired.imag)
        axes[1, 1].set_title('imag desired random target')
        axes[2, 0].imshow(np.abs(mat_best))
        axes[2, 0].set_title('abs best optimized')
        axes[2, 1].imshow(np.abs(self.U_desired))
        axes[2, 1].set_title('abs desired random target')
        axes[3, 0].imshow(np.angle(mat_best))
        axes[3, 0].set_title('phase best optimized')
        axes[3, 1].imshow(np.angle(self.U_desired))
        axes[3, 1].set_title('phase desired random target')
        fig.suptitle(f'N={self.N}, M={self.M}, best_cost={best_cost:.3f}')
        fig.show()

    def run(self):
        self.optimizer = MyPSOOptimizer(self.cost_function, n_pop=40, n_var=self.N*self.M,
                                   n_iterations=1000,
                                   w=1, wdamp=0.99, c1=1.5, c2=2,
                                   timeout=30 * 60,
                                   stop_early=True, stop_after_n_const_iter=30,
                                   vary_popuation=True, reduce_at_iterations=(2,),
                                   quiet=True)

        self.optimizer.optimize()

        return np.abs(self.optimizer.best_cost)


def single_run(N, M=None):
    s = NsqrDOFSimulation(N=N, M=M)
    s.run()
    s.show_results()
    plt.show()


def statistics(mean_on_N=5, Ns_DOF=(3, 5, 10, 12), M_factors=(1, 2, 5)):
    for N in Ns_DOF:
        for M_factor in M_factors:
            costs = np.zeros(mean_on_N)
            for i in range(mean_on_N):
                s = NsqrDOFSimulation(N=N, M=N*M_factor)
                costs[i] = s.run()

            print(f"N={N}, M={N*M_factor}, cost = {costs.mean()} +- {costs.std()}")


if __name__ == "__main__":
    statistics(10, Ns_DOF=(3, 5, 10, 12), M_factors=(1, 2, 5))
    # single_run(N=10, M=100)
