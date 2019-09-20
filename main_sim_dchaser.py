import os
import time
import argparse

import numpy as np
from src import utils
from src import DChaserSim

def main():
    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)

    chaser_sim = DChaserSim(ARGS.num_particles, ARGS.num_targets,ARGS.steps,ARGS.dt,ARGS.prefix)
    data_all = chaser_sim.simulation(8)
    # data_all = utils.run_simulation_d(chaser_sim, ARGS.instances, ARGS.processes,
    #                                 ARGS.batch_size)
    position_data_all     = data_all[0]
    velocity_data_all     = data_all[1]
    acceleration_data_all = data_all[2]
    edge_data_all         = data_all[3]

    d_position_velocity_data_all = data_all[4]
    d_n_view_data_all            = data_all[5]
    d_edge_data_all              = data_all[6]

    d_next_position_velocity_data_all = data_all[7]
    d_next_acceleration_data_all      = data_all[8]

    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_position.npy'), position_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_velocity.npy'), velocity_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_acceleration.npy'), acceleration_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_edge.npy'), edge_data_all)

    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_d_pos_vel.npy'), d_position_velocity_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_d_n_view.npy'), d_n_view_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_d_edge.npy'),d_edge_data_all)
    print(d_edge_data_all)
    print(edge_data_all)

    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_d_next_pos_vel.npy'),
                                                d_next_position_velocity_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_d_next_acceleration.npy'),
                                                d_next_acceleration_data_all)


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
