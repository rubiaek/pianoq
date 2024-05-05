import numpy as np
import matplotlib.pyplot as plt


class RandomPCCSimulation(object):
    N_repeats = 200
    MAX_Nmodes = 300

    def __init__(self):
        self.random_pccs = []
        self.random_pccs_std = []

    def run(self):
        for Nmodes in range(1, self.MAX_Nmodes):
            pcc, pcc_std = self.get_random_pcc(Nmodes)
            self.random_pccs.append(pcc)
            self.random_pccs_std.append(pcc_std)

        self.plot_result()

    def get_random_pcc(self, Nmodes):
        pccs = []

        for i in range(self.N_repeats):
            pcc = self.get_correlation(self.sample(Nmodes), self.sample(Nmodes))
            pccs.append(pcc)

        return np.mean(pccs), np.std(pccs)

    def sample(self, Nmodes):
        """ Get random sample of mode distribution """
        # Following https://stackoverflow.com/questions/29187044/generate-n-random-numbers-within-a-range-with-a-constant-sum/29187687
        V = np.random.uniform(0, 1, Nmodes)
        U = -np.log(V)
        P = U / U.sum()

        return P

    def get_correlation(self, V1, V2):
        numerator = np.sum((V1 - np.mean(V1)) * (V2 - np.mean(V2)))

        A = (V1 - np.mean(V1)) ** 2
        B = (V2 - np.mean(V2)) ** 2
        denumerator = np.sqrt(np.sum(A) * np.sum(B))
        dist_ncc = numerator / denumerator
        return dist_ncc

    def plot_result(self):
        fig, ax = plt.subplots()
        ax.errorbar(range(1, self.MAX_Nmodes), self.random_pccs, yerr=self.random_pccs_std, fmt='.--')
        ax.set_xlabel('Nmodes')
        ax.set_ylabel('PCC')
        fig.show()


if __name__ == "__main__":
    rps = RandomPCCSimulation()
    # pcc, pcc_std = rps.get_random_pcc(2000)
    # print(f'pcc, pcc_std: {pcc:.5f}, {pcc_std:.5f}')
    rps.run()
    plt.show()

