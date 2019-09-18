import numpy as np
from src.agents.agent import Agent

class ChaserAgent(Agent):
    """A Particle agent that chases another Particle"""

    def __init__(self, position, velocity=None, acceleration=None,
                 ndim=None, max_speed=None, max_acceleration=None,
                 vision_radius=None, vision_angle=None):
        super().__init__(position, velocity, acceleration, ndim, max_speed, max_acceleration,
                                vision_radius, vision_angle)

        self._targets = []

    @property
    def targets(self):
        return self._targets

    def add_target(self, new_target):
        if new_target:
            if not isinstance(new_target, Agent):
                raise ValueError('new_target must be a Particle')

            self._targets.append(new_target)

    def decide(self):
        displacement = np.zeros(self.ndim)
        for target in self.targets:
            if self.can_see(target):
                displacement += target.position - self.position

        self.acceleration = displacement


# class DynamicalChaser(Agent):
#     """An Agent that chases its neighbors."""
#     pass
