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
    def __init__(self, position, agent_id, params, velocity=None, acceleration=None,
                agent_type=None):
        """
        Create a boid with essential attributes.
        `ndim`: dimension of the space it resides in.
        `vision`: the visual range.
        `anticipation`: range of anticipation for its own motion.
        `comfort`: distance the agent wants to keep from other objects.
        `max_speed`: max speed the agent can achieve.
        `max_acceleratoin`: max acceleration the agent can achieve.
        """
        super().__init__(position, velocity, acceleration, params['n_dim'],
                    params['max_speed'],params['max_acceleration'])
        # it is assumed that id given is unique
        self.agent_id      = agent_id
        self.agent_type    = agent_type
        self.size          = float(params['agent_size']) if params['agent_size'] !=0 else 0.
        self.vision_radius = float(params['vision_radius']) if params['vision_radius'] != 0 else np.inf
        self.vision_angle  = float(params['vision_angle']) if params['vision_angle'] != 0 else np.pi
        self.ego_view      = params['ego_view']
        self.padding_size  = params['padding_size']

        self.neighbors = []
        self.obstacles = []
        self.goals     = []
        self._targets = []

        # self.n_goals      = None
        # self.n_agents     = None
        # self.n_obstacles  = None

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
        self.computeRotationTransformation()
        # NOTE: only implemented for 2d data
        position_data     = np.zeros((self.padding_size,2))
        velocity_data     = np.zeros((self.padding_size,2))
        acceleration_data = np.zeros((self.padding_size,2))
        edge_data         = np.zeros((self.padding_size-1))

        counter = 0
        if self.ego_view == True:
            position_data[counter,:]     =  np.zeros(2)
            velocity_data[counter,1]     =  np.linalg.norm(self.velocity)
            acceleration_data[counter,:] =  np.matmul(self.rotation_matrix, self.acceleration)
        else:
            position_data[counter,:]     =  self.position.copy()
            velocity_data[counter,:]     =  self.velocity.copy()
            acceleration_data[counter,:] =  self.acceleration.copy()

        counter += 1

        for goal in self.goals:
            t_position, t_velocity,t_acceleration = self.transformToView2D(
                                                    goal.position, np.zeros(2),np.zeros(2))
            position_data[counter,:]     =  t_position
            velocity_data[counter,:]     =  t_velocity
            acceleration_data[counter,:] =  t_acceleration
            edge_data[counter-1]           =  self.edgeType(self.agent_type,ObjectType.GOAL)
            counter += 1

        for obstacle in self.obstacles:
            t_position, t_velocity,t_acceleration = self.transformToView2D(
                                                    obstacle.position, np.zeros(2),np.zeros(2))
            position_data[counter,:]     =  t_position
            velocity_data[counter,:]     =  t_velocity
            acceleration_data[counter,:] =  t_acceleration
            edge_data[counter-1]           =  self.edgeType(self.agent_type,ObjectType.OBSTACLE)
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
                    edge_data[counter-1]   =  self.edgeType(self.agent_type,ObjectType.CHASER)
            else:
                edge_data[counter-1]       =  self.edgeType(self.agent_type,self.agent.agent_type)
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

    def computeRotationTransformation(self):

        if np.linalg.norm(self.velocity) > 0.001:
            theta_unit_vector =  self.velocity/np.linalg.norm(self.velocity)
            self.theta             =  np.arccos(theta_unit_vector[0]) * \
                                            np.sign(theta_unit_vector[1])
        else:
            self.theta             = 0.


        self.rotation_matrix       = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                                            [np.sin(self.theta), np.cos(self.theta)]])
        self.transformation_matrix = np.array([[np.sin(self.theta),np.cos(self.theta)],
                                            [np.cos(self.theta),-np.sin(self.theta)]])
        self.prev_position         = self.position.copy()

    # NOTE: only works with 2d environments
    def transformToView2D(self,position,velocity,acceleration):
        ''' transforms the other agents state info w.r.t. current agents view '''
        if self.ego_view == True:
            # always y-axis is denoted by the heading direction
            t_position     = np.matmul(self.transformation_matrix, position - self.prev_position)
            t_velocity     = np.matmul(self.rotation_matrix, velocity)# - self.velocity)
            t_acceleration = np.matmul(self.rotation_matrix, acceleration)# - self.acceleration)
            return t_position, t_velocity, t_acceleration

        else:
            return position, velocity, acceleration

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
