import numpy as np
from enum import Enum
from src.agents.particle import Particle

class ObjectType(Enum):
    GOAL     = 0
    OBSTACLE = 1
    CHASER   = 2
    BOIDS    = 3
    VICSEK   = 4

class DAgent(Particle):
    def __init__(self, position, agent_id,velocity=None, acceleration=None,
                ndim=None, max_speed=None, max_acceleration=None,
                size=None, vision_radius=None, vision_angle = None,agent_type=None,
                padding_size = 50):
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
        # it is assumed that id given is unique
        self.agent_id      = id
        self.agent_type    = agent_type
        self.size          = float(size) if size else 0.
        self.vision_radius = float(vision_radius) if vision_radius else np.inf
        self.vision_angle  = float(vision_angle) if vision_angle else np.pi

        self.neighbors = []
        self.obstacles = []
        self.goals     = []
        self._targets = []

        # self.n_goals      = None
        # self.n_agents     = None
        # self.n_obstacles  = None
        self.padding_size = padding_size

    @property
    def targets(self):
        return self._targets

    def add_target(self, new_target):
        if new_target:
            if not isinstance(new_target, Particle):
                raise ValueError('new_target must be a Particle')

        self._targets.append(new_target)

    def isTarget(self, agent):
        ''' returns true if agent is target'''
        for t_agent in self.targets:
            if t_agent == agent:
                return True

        return False

    def canSee(self, other):
        """Whether the boid can see the other."""
        return self.distance(other) < self.vision_radius
        # TODO: implement cone based vision

    def inViewData(self):
        ''' Return the state and edge information of other objects in view w.r.t to
        current agent '''
        # NOTE: only implemented for 2d data
        position_data     = np.zeros((self.padding_size,2))
        velocity_data     = np.zeros((self.padding_size,2))
        acceleration_data = np.zeros((self.padding_size,2))
        edge_data         = np.zeros((self.padding_size-1))

        counter = 0
        position_data[counter,:]     =  np.zeros(2)
        velocity_data[counter,:]     =  np.zeros(2)
        acceleration_data[counter,:] =  np.zeros(2)
        counter += 1

        for goal in self.goals:
            t_position, t_velocity,t_acceleration = self.transformToView2D(
                                                    goal.position, np.zeros(2),np.zeros(2))
            position_data[counter,:]     =  t_position
            velocity_data[counter,:]     =  t_velocity
            acceleration_data[counter,:] =  t_acceleration
            edge_data[counter]           =  self.edgeType(self.agent_type,ObjectType.GOAL)
            counter += 1

        for obstacle in self.obstacles:
            t_position, t_velocity,t_acceleration = self.transformToView2D(
                                                    obstacle.position, np.zeros(2),np.zeros(2))
            position_data[counter,:]     =  t_position
            velocity_data[counter,:]     =  t_velocity
            acceleration_data[counter,:] =  t_acceleration
            edge_data[counter]           =  self.edgeType(self.agent_type,ObjectType.OBSTACLE)
            counter += 1

        for agent in self.neighbors:
            t_position, t_velocity,t_acceleration = self.transformToView2D(
                                                    agent.position, agent.velocity,
                                                    agent.acceleration)
            position_data[counter,:]     =  t_position
            velocity_data[counter,:]     =  t_velocity
            acceleration_data[counter,:] =  t_acceleration
            if ((agent.agent_type ==  ObjectType.CHASER)
                                and (self.agent_type == ObjectType.CHASER)):
                if self.isTarget(agent):
                    edge_data[counter]   =  self.edgeType(self.agent_type,ObjectType.CHASER)
            else:
                edge_data[counter]       =  self.edgeType(self.agent_type,self.agent.agent_type)
            counter += 1

        return position_data, velocity_data, acceleration_data, edge_data, counter

    def edgeType(self,object_type_target, object_type_source):
        ''' Edge matrix '''
        edge_type = np.array([[0, 0, 2, 4, 8],
                                [0, 0, 3, 5, 9],
                                [0, 0, 1, 1, 1],
                                [0, 0, 3, 6, 10],
                                [0, 0, 3, 7, 11]])

        return edge_type[object_type_source.value,object_type_target.value]

    # NOTE: only works with 2d environments
    def transformToView2D(self,position,velocity,acceleration):
        ''' transforms the other agents state info w.r.t. current agents view '''
        # always y-axis is denoted by the heading direction
        if np.linalg.norm(self.velocity) > 0.001:
            theta_unit_vector =  self.velocity/np.linalg.norm(self.velocity)
            theta             =  np.arccos(theta_unit_vector[0]) * \
                                            np.sign(theta_unit_vector[1])
        else:
            theta             = 0.


        rotation_matrix       = np.array([[np.cos(theta), -np.sin(theta)],
                                            [np.sin(theta), np.cos(theta)]])
        transformation_matrix = np.array([[np.sin(theta),np.cos(theta)],
                                            [np.cos(theta),-np.sin(theta)]])
        t_position     = np.matmul(transformation_matrix, position - self.position)
        t_velocity     = np.matmul(rotation_matrix, velocity - self.velocity)
        t_acceleration = np.matmul(rotation_matrix, acceleration - self.acceleration)

        return t_position, t_velocity, t_acceleration

    def observe(self,environment):
        """Observe the population and take note of neighbors."""
        # self.n_goals     = len(environment.goals)
        # self.n_agents    = len(environment.population)
        # self.n_obstacles = len(environment.obstacles)

        self.neighbors = [other for other in environment.population
                          if self.canSee(other) and id(other) != id(self)]
        # To simplify computation, it is assumed that agent is aware of all
        # obstacles including the boundaries. In reality, the agent is only
        # able to see the obstacle when it is in visual range. This doesn't
        # affect agent's behavior, as agent only reacts to obstacles when in
        # proximity, and no early planning by the agent is made.
        self.obstacles = [obstacle for obstacle in environment.obstacles
                          if self.canSee(obstacle)]
        self.goals     = environment.goals
