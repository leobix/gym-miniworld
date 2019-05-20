import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..params import DEFAULT_PARAMS

class FourRoomsAddict(MiniWorldEnv):
    """
    Classic four rooms environment
    """

    def __init__(self, forward_step=0.7, turn_step=45, **kwargs):

        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', forward_step)
        params.set('turn_step', turn_step)
        # Allow only the movement actions

        super().__init__(
            max_episode_steps=500,
            params=params,
            **kwargs
        )

        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        # Top-left room
        room0 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=1 , max_z=7
        )
        # Top-right room
        room1 = self.add_rect_room(
            min_x=1, max_x=7,
            min_z=1, max_z=7
        )
        # Bottom-right room
        room2 = self.add_rect_room(
            min_x=1 , max_x=7,
            min_z=-7, max_z=-1
        )
        # Bottom-left room
        room3 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=-7, max_z=-1
        )

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=3, max_z=5, max_y=2.2)
        self.connect_rooms(room1, room2, min_x=3, max_x=5, max_y=2.2)
        self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2.2)
        self.connect_rooms(room3, room0, min_x=-5, max_x=-3, max_y=2.2)

        self.box = self.place_entity(Box(color='red'))
        self.badbox = self.place_entity(Box(color='yellow', size=[0.8, 1.2, 0.5]))

        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward -= self._reward()
            done = True

        if self.near(self.badbox):
            reward += self._reward()
            done = True

        return obs, reward, done, info
