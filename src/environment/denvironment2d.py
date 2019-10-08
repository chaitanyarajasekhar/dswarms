import numpy as np

from src.environment.environment2d import Environment2D
from ..agents import DAgent

class DEnvironment2D(Environment2D):
    ''' custom update for decentralized agent '''
    def __init__(self,boundary=None):
        super().__init__(boundary)

    def add_agent(self, agent):
        if not isinstance(agent, DAgent):
            raise ValueError('agent must be an instance of Agent')

        if agent.ndim != 2:
            raise ValueError('position space of agent must be 2D')
        self.population.append(agent)

    def update(self, dt, test = False):
        """
        Update the state of environment for one time step dt, during which the
        boids move.
        """
        # TODO: implement for test data later
        previous_step_data = []
        previous_step_data.append([])
        previous_step_data.append([])
        previous_step_data.append([])

        current_step_data  = []
        current_step_data.append([])
        current_step_data.append([])

        # if test = True:
        #     population = self.population[0]
        # else:
        #     population = self.population

        for agent in self.population:
            agent.observe(self)
            # save the current view data
            data = agent.inViewData()
            position_data     = data[0]
            velocity_data     = data[1]
            previous_step_data[0].append(np.concatenate((position_data,velocity_data),axis=1))
            # not using acceleration data
            # acceleration_data = data[2]
            previous_step_data[1].append(data[4]) # counter i.e. agents in view
            edge_data         = data[3]
            previous_step_data[2].append(edge_data)

            agent.decide(self.goals)
        # Hold off moving agents until all have made decision.
        # This ensures synchronous update.
        for agent in self.population:
            agent.move(dt)

        for agent in self.population:
            pos,vel,acc = agent.transformToView2D(agent.position.copy(),
                                agent.velocity.copy(),agent.acceleration.copy())
            current_step_data[0].append(np.concatenate((pos,vel)))
            current_step_data[1].append(acc)

        return previous_step_data, current_step_data
