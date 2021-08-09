import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx


def main():
    # Set-up hyperparameters
    # NUM_OF_ITERATIONS, 'nPop', 20, 'wdamp', 0.99, 'c1', 1.5, 'c2', 2
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
    # Perform optimization
    best_cost, best_pos = optimizer.optimize(fx.sphere, iters=100)


if __name__ == "__main__":
    main()
