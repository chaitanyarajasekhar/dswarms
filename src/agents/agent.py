import numpy as np
from src.agents.particle import Particle

class Agent(Particle):
    def __init__(self, position, velocity=None, acceleration=None,
                 ndim=None, max_speed=None, max_acceleration=None,
                 size=None, vision_radius=None, vision_angle = None):
        """
        Create a boid with essential attributes.
        `ndim`: dimension of the space it resides in.
        `vision`: the visual range.
        `anticipation`: range of anticipation for its own motion.
        `comfort`: distance the agent wants to keep from other objects.
        `max_speed`: max speed the agent can achieve.
        `max_acceleratoin`: max acceleration the agent can achieve.
        """
        super().__init__(position, velocity, acceleration, ndim, max_speed, max_acceleration)

        self.size          = float(size) if size else 0.
        self.vision_radius = float(vision) if vision_radius else np.inf
        self.vision_angle  = float(vision) if vision_angle else np.pi

        self.neighbors = []
        self.obstacles = []

    def can_see(self, other):
        """Whether the boid can see the other."""
        return self.distance(other) < self.vision_radius
        # TODO: implement cone based vision

    def inViewAgentsStates(self):
        # for other_agent in self.neighbors:
        #     position_agent, velocity_agent, acceleration_agent =
        #                 self.transformToMyView(other_agent.)
        pass


    def transformToMyView(self):
        pass

    def observe(self):
        """Observe the population and take note of neighbors."""
        self.neighbors = [other for other in environment.population
                          if self.can_see(other) and id(other) != id(self)]
        # To simplify computation, it is assumed that agent is aware of all
        # obstacles including the boundaries. In reality, the agent is only
        # able to see the obstacle when it is in visual range. This doesn't
        # affect agent's behavior, as agent only reacts to obstacles when in
        # proximity, and no early planning by the agent is made.
        self.obstacles = [obstacle for obstacle in environment.obstacles
                          if self.can_see(obstacle)]
