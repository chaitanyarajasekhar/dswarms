import numpy as np

from src.agents import DChaserAgent
from src.environment import DEnvironment2D

class DChaserSim:
    def __init__(self,num_agents,num_targets,steps,dt,prefix=None):
        self.num_targets = num_targets
        self.num_agents  = num_agents
        self.dt          = dt
        self.steps       = steps
        self.prefix      = True if prefix == 'test' else False

    def createChasers(self):
        """
        Create n particle chasers, each with m targets randomly selected with in the group.
        n = num_agents, m = num_targets
        """
        if self.num_agents < 1:
            raise ValueError('num_agents must be a positive integer')

        if self.num_targets  == 'x':
            self.num_targets = np.random.randint(1, self.num_agents)
        elif self.num_targets == 'y':
            pass
        elif int(self.num_targets) < 1 or int(self.num_targets) > self.num_agents - 1:
            raise ValueError('num_targets must be a positive integer less than n')

        prev = None
        particles = []
        for _ in range(self.num_agents):
            r = 20
            theta = np.random.rand() * 2 * np.pi
            x, y = r * np.cos(theta), r * np.sin(theta)
            v = np.random.uniform(-2, 2, 2)

            particles.append(DChaserAgent((x, y), v, ndim=2, max_speed=10, max_acceleration=10))

        edges = np.zeros((self.num_agents, self.num_agents))
        particle_idxs = np.arange(self.num_agents)
        for i, p in enumerate(particles):
            if self.num_targets == 'y':
                k = np.random.randint(1, self.num_agents)
            else:
                k = int(self.num_targets)

            for j in np.random.choice(particle_idxs[particle_idxs != i], k, replace=False):
                edges[j, i] = 1  # j is i's target, thus j influences i through edge j->i.
                p.add_target(particles[j])

        return particles, edges

    def simulation(self,_):
        np.random.seed()

        agents, edges = self.createChasers()

        env = DEnvironment2D()

        for agent in agents:
            env.add_agent(agent)

        # centralized_data
        position_data     = []
        velocity_data     = []
        acceleration_data = []

        d_position_velocity_data     = []
        d_n_view_data                = []
        d_edge_data                  = []

        d_next_position_velocity_data = []
        d_next_acceleration_data      = []

        for _ in range(self.steps):
            step_position     = []
            step_velocity     = []
            step_acceleration = []
            for agent in agents:
                step_position.append(agent.position.copy())
                step_velocity.append(agent.velocity.copy())
                step_acceleration.append(agent.acceleration.copy())

            previous_step_data, current_step_data = env.update(self.dt)
            d_position_velocity_data.append(previous_step_data[0])
            d_n_view_data.append(previous_step_data[1])
            d_edge_data.append(previous_step_data[2])

            d_next_position_velocity_data.append(current_step_data[0])
            d_next_acceleration_data.append(current_step_data[1])

            position_data.append(step_position)
            velocity_data.append(step_velocity)
            acceleration_data.append(step_acceleration)

        return position_data, velocity_data, acceleration_data, edges, d_position_velocity_data,\
                d_n_view_data, d_edge_data, d_next_position_velocity_data, d_next_acceleration_data
