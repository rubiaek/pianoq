from pianoq.lab.piano_optimization import PianoOptimization


def main():
    n = 3
    for i in range(3):
        po = PianoOptimization(saveto_path=None)
        # po.optimize_my_pso(n_pop=10, n_iterations=10, stop_after_n_const_iters=25, reduce_at_iterations=(1,))
        po.optimize_my_pso(n_pop=25, n_iterations=150, stop_after_n_const_iters=20, reduce_at_iterations=(4, 10))
        po.close()


if __name__ == "__main__":
    main()
