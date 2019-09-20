import numpy as np
from src.agents.dagent import DAgent, ObjectType


class DChaserAgent(DAgent):
    def __init__(self, position, agent_id, velocity=None, acceleration=None,
                ndim=None, max_speed=None, max_acceleration=None,
                size=None, vision_radius=None, vision_angle = None,
                agent_type = ObjectType.CHASER):
        '''Decentralized Chaser Agent'''
        super().__init__(position, agent_id, velocity, acceleration, ndim,
                    max_speed, max_acceleration,size, vision_radius, vision_angle,
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
