import numpy as np
from src.agents import ChaserAgent

class ChaserSim:
    def __init__(self,num_agents,num_targets,steps,dt):
        self.num_targets = num_targets
        self.num_agents  = num_agents
        self.dt          = dt
        self.steps       = steps

    def create_chasers(self):
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

            particles.append(ChaserAgent((x, y), v, ndim=2, max_speed=10, max_acceleration=10))

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


    # def chasers_edges(n):
    #     """
    #     Edges for a list of chaser particles in which each agent chases its predecessor in the list.
    #     A 1 at Row i, Column j means Particle i influences Particle j. No influence
    #     is represented by 0.
    #     """
    #     matrix = np.zeros((n, n), dtype=int)
    #     for i in range(n):
    #         matrix[i, (i+1) % n] = 1

    #     return matrix


    def simulation(self,_):
        np.random.seed()

        particles, edges = self.create_chasers()

        position_data = []
        velocity_data = []

        for _ in range(self.steps):
            step_position = []
            step_velocity = []
            for p in particles:
                step_position.append(p.position.copy())
                step_velocity.append(p.velocity.copy())

                p.decide()

            for p in particles:
                p.move(self.dt)

            position_data.append(step_position)
            velocity_data.append(step_velocity)

        return position_data, velocity_data, edges
