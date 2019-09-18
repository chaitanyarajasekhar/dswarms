import os
import time
import argparse

import numpy as np
from src import utils
from src import ChaserSim

def main():
    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)

    chaser_sim = ChaserSim(ARGS.num_particles, ARGS.num_targets,ARGS.steps,ARGS.dt)
    position_data_all, velocity_data_all, edge_data_all = utils.run_simulation(
                                                                            chaser_sim,
                                                                            ARGS.instances,
                                                                            ARGS.processes,
                                                                            ARGS.batch_size)

    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_position.npy'), position_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_velocity.npy'), velocity_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_edge.npy'), edge_data_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-particles', '-n', type=int, default=5,
                        help='number of particles')
    parser.add_argument('--num-targets', '-m', type=str, default=1,
                        help="number of targets for each particle\n"
                             "use 'x' for a random number\n"
                             "use 'y' for a different random number for each agent")
    parser.add_argument('--instances', type=int, default=1000,
                        help='number of instances to run')
    parser.add_argument('--steps', type=int, default=50,
                        help='number of time steps per simulation')
    parser.add_argument('--dt', type=float, default=0.3,
                        help='unit time step')
    parser.add_argument('--save-dir', type=str,
                        help='name of the save directory')
    parser.add_argument('--prefix', type=str, default='',
                        help='prefix for save files')
    parser.add_argument('--save-edges', action='store_true', default=False,
                        help='Deprecated. Now edges are always saved.')
    parser.add_argument('--processes', type=int, default=1,
                        help='number of parallel processes')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='number of simulation instances for each process')

    ARGS = parser.parse_args()

    ARGS.save_dir = os.path.expanduser(ARGS.save_dir)

    main()
