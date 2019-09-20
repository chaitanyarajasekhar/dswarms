import time
import functools
import multiprocessing


def run_simulation(simulation_object, instances, processes=1, batch=100, silent=False):
    pool = multiprocessing.Pool(processes=processes)
    position_data_all     = []
    velocity_data_all     = []
    edge_data_all         = []
    acceleration_data_all = []

    remaining_instances = instances

    prev_time = time.time()
    while remaining_instances > 0:
        n = min(remaining_instances, batch)
        data_pool = pool.map(simulation_object.simulation, range(n))

        position_pool, velocity_pool, acceleration_pool, edge_pool = zip(*data_pool)

        remaining_instances -= n
        if not silent:
            print('Simulation {}/{}... {:.1f}s'.format(instances - remaining_instances,
                                                       instances, time.time()-prev_time))
        prev_time = time.time()

        position_data_all.extend(position_pool)
        velocity_data_all.extend(velocity_pool)
        acceleration_data_all.extend(acceleration_pool)
        edge_data_all.extend(edge_pool)

    return position_data_all, velocity_data_all, acceleration_data_all, edge_data_all

def run_simulation_d(simulation_object, instances, processes=1, batch=100, silent=False):
    pool = multiprocessing.Pool(processes=processes)
    position_data_all     = []
    velocity_data_all     = []
    edge_data_all         = []
    acceleration_data_all = []

    d_position_velocity_data_all = []
    d_n_view_data_all            = []
    d_edge_data_all              = []

    d_next_position_velocity_data_all  = []
    d_next_acceleration_data_all       = []

    remaining_instances = instances

    prev_time = time.time()
    while remaining_instances > 0:
        n = min(remaining_instances, batch)
        data_pool = pool.map(simulation_object.simulation, range(n))

        print(type(data_pool))

        # data_pool                     = zip(*data_pool)
        # position_pool                 =
        # velocity_pool                 =
        # acceleration_pool             =
        # edge_pool                     =
        # d_position_velocity_pool      =
        # d_n_view_pool                 =
        # d_edge_pool                   =
        # d_next_position_velocity_pool =
        # d_next_acceleration_pool      =

        remaining_instances -= n
        if not silent:
            print('Simulation {}/{}... {:.1f}s'.format(instances - remaining_instances,
                                                       instances, time.time()-prev_time))
        prev_time = time.time()

        # position_data_all.extend(position_pool)
        # velocity_data_all.extend(velocity_pool)
        # acceleration_data_all.extend(acceleration_pool)
        # edge_data_all.extend(edge_pool)
        #
        # d_position_velocity_data_all.append(d_position_velocity_pool)
        # d_n_view_data_all.append(d_n_view_pool)
        # d_edge_data_all.append(d_edge_pool)
        #
        # d_next_position_velocity_data_all.append(d_next_position_velocity_pool)
        # d_next_acceleration_data_all.append(d_next_acceleration_pool)

    return position_data_all, velocity_data_all, acceleration_data_all, edge_data_all,\
            d_position_velocity_data_all, d_n_view_data_all, d_edge_data_all,\
            d_next_position_velocity_data_all, d_next_acceleration_data_all
