import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from gym import spaces
from ..params import DEFAULT_PARAMS


class TMazeAddict(MiniWorldEnv):
    """
    Two hallways connected in a T-junction
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=280,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        room1 = self.add_rect_room(
            min_x=-1, max_x=8,
            min_z=-2, max_z=2
        )
        room2 = self.add_rect_room(
            min_x=8, max_x=12,
            min_z=-8, max_z=8
        )
        self.connect_rooms(room1, room2, min_z=-2, max_z=2)

        # Add a box at a random end of the hallway
        self.box = Box(color='red')
        if self.rand.bool():
            self.place_entity(self.box, room=room2, min_z=room2.max_z - 2)
        else:
            self.place_entity(self.box, room=room2, max_z=room2.min_z + 2)

        # Choose a random room and position to spawn at
        self.place_agent(
            dir=self.rand.float(-math.pi/4, math.pi/4),
            room=room1
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward -= self._reward()
            done = True

        if self.box.pos[2] > 0:
            if self.agent.pos[2] - 1 < - self.box.pos[2]:
                reward += self._reward()
                done = True
        else:
            if self.agent.pos[2] + 1 > - self.box.pos[2] :
                reward += self._reward()
                done = True

        return obs, reward, done, info


class TMazeAddict2(MiniWorldEnv):
    """
    Two hallways connected in a T-junction
    """

    def __init__(self, forward_step=0.7, turn_step=45, **kwargs):

        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', forward_step)
        params.set('turn_step', turn_step)
        # Allow only the movement actions

        super().__init__(
            max_episode_steps=280,
            params=params,
            **kwargs
        )

        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        room1 = self.add_rect_room(
            min_x=-1, max_x=8,
            min_z=-2, max_z=2
        )
        room2 = self.add_rect_room(
            min_x=8, max_x=12,
            min_z=-8, max_z=8
        )
        self.connect_rooms(room1, room2, min_z=-2, max_z=2)

        # Add a box at a random end of the hallway
        self.box = Box(color='red')
        if self.rand.bool():
            self.place_entity(self.box, room=room2, min_z=room2.max_z - 2)
        else:
            self.place_entity(self.box, room=room2, max_z=room2.min_z + 2)

        # Choose a random room and position to spawn at
        self.place_agent(
            dir=self.rand.float(-math.pi/4, math.pi/4),
            room=room1,
            min_x=6
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward -= self._reward()
            done = True

        if self.box.pos[2] > 0:
            if self.agent.pos[2] - 1 < - self.box.pos[2]:
                reward += self._reward()
                done = True
        else:
            if self.agent.pos[2] + 1 > - self.box.pos[2] :
                reward += self._reward()
                done = True

        return obs, reward, done, info
