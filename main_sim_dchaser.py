import os
import time
import argparse
import json

import numpy as np
from src import utils
from src import DChaserSim

def main():

    with open(ARGS.config) as f:
        params = json.load(f)

    params.update({"save_dir": os.path.expanduser(params['save_dir'])})

    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])

    # chaser_sim = DChaserSim(ARGS.num_particles, ARGS.num_targets,ARGS.steps,ARGS.dt,ARGS.prefix)
    chaser_sim =  chaser_sim = DChaserSim(params)
    # data_all = chaser_sim.simulation(8)
    data_all = utils.run_simulation_d(chaser_sim, params['instances'], params['processes'],
                                    params['batch_size'])
    position_data_all     = data_all[0]
    velocity_data_all     = data_all[1]
    acceleration_data_all = data_all[2]
    edge_data_all         = data_all[3]

    d_position_velocity_data_all = data_all[4]
    d_n_view_data_all            = data_all[5]
    d_edge_data_all              = data_all[6]

    d_next_position_velocity_data_all = data_all[7]
    d_next_acceleration_data_all      = data_all[8]

    np.save(os.path.join(params['save_dir'], params['prefix']+'_position.npy'), position_data_all)
    np.save(os.path.join(params['save_dir'], params['prefix']+'_velocity.npy'), velocity_data_all)
    np.save(os.path.join(params['save_dir'], params['prefix']+'_acceleration.npy'), acceleration_data_all)
    np.save(os.path.join(params['save_dir'], params['prefix']+'_edge.npy'), edge_data_all)

    np.save(os.path.join(params['save_dir'], params['prefix']+'_d_pos_vel.npy'), d_position_velocity_data_all)
    np.save(os.path.join(params['save_dir'], params['prefix']+'_d_n_view.npy'), d_n_view_data_all)
    np.save(os.path.join(params['save_dir'], params['prefix']+'_d_edge.npy'),d_edge_data_all)

    np.save(os.path.join(params['save_dir'], params['prefix']+'_d_next_pos_vel.npy'),
                                                d_next_position_velocity_data_all)
    np.save(os.path.join(params['save_dir'], params['prefix']+'_d_next_acceleration.npy'),
                                                d_next_acceleration_data_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num-targets', '-m', type=str, default=1,
    #                     help="number of targets for each particle\n"
    #                          "use 'x' for a random number\n"
    #                          "use 'y' for a different random number for each agent")
    parser.add_argument('--config', type=str,
                        help='config file location')
    ARGS = parser.parse_args()

    main()
