from pianoq.lab.PianoOptimization import PianoOptimization


def main():
    po = PianoOptimization(saveto_path=None)
    # po.optimize_my_pso(n_pop=10, n_iterations=10, stop_after_n_const_iters=25, reduce_at_iterations=(1,))
    po.optimize_my_pso(n_pop=25, n_iterations=150, stop_after_n_const_iters=25, reduce_at_iterations=(4, 10))
    po.close()


if __name__ == "__main__":
    main()
