import numpy as np
from src.agents.dagent import DAgent, ObjectType


class DChaserAgent(DAgent):
    def __init__(self, position, agent_id,params, velocity=None, acceleration=None,
                agent_type = ObjectType.CHASER):
        '''Decentralized Chaser Agent'''
        super().__init__(position, agent_id, params, velocity, acceleration,
                    agent_type)

    #     self._targets = []
    #
    # @property
    # def targets(self):
    #     return self._targets
    #
    # def add_target(self, new_target):
    #     if new_target:
    #         if not isinstance(new_target, DAgent):
    #             raise ValueError('new_target must be a Particle')
    #
    #         self._targets.append(new_target)

    # TODO: add obstacle avoidance to the chaser model
    def decide(self,_):
        displacement = np.zeros(self.ndim)
        for target in self.targets:
            if self.canSee(target):
                displacement += target.position - self.position

        self.acceleration = displacement
