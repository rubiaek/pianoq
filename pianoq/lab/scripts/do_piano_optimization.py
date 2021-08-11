from pianoq.lab.PianoOptimization import PianoOptimization


def main():
    po = PianoOptimization(saveto_path=None)
    po.optimize_my_pso(n_pop=20, n_iterations=50, stop_after_n_const_iters=25, reduce_at_iterations=(7,))
    po.close()


if __name__ == "__main__":
    main()
