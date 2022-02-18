from collections import deque
from typing import Any, NamedTuple
from gym import spaces
import copy

import numpy as np

import metaworld
import random


class SequentialTaskWrapper(object):
    def __init__(self, env_list, max_steps=500):
        self.env_list = env_list
        self.n_envs = len(env_list)
        self.current_env = self._pick_env()
        self.current_step = 0
        self.max_steps = max_steps
        self.done = False

    @property
    def action_space(self):
        return self.current_env.action_space

    @property
    def observation_space(self):
        return self.current_env.observation_space

    def reset(self):
        self.current_env = self._pick_env()
        obs = self.current_env.reset()
        self.current_step = 0
        self.done = False
        return obs

    def step(self, action):
        next_obs, rew, done, info = self.current_env.step(action)
        self.current_step += 1
        assert not self.done
        done = done or self.current_step > self.max_steps > 0
        self.done = done
        return next_obs, rew, done, info

    def _pick_env(self):
        self.task_id = np.random.randint(low=0, high=self.n_envs, size=(1, )).item()
        return self.env_list[self.task_id]

    def seed(self, seed):
        for env in self.env_list:
            env.seed(seed)
    
    def close(self):
        for env in self.env_list:
            env.close()


class PixelWrapper:
    def __init__(self,
                 env,
                 height=96,
                 width=96,
                 n_channels=3,
                 camera_name='corner3'):
        self.env = env
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.camera_name = camera_name

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=[self.height, self.width, self.n_channels],
            dtype=np.uint8)
        self.action_space = env.action_space

    def reset(self):
        _ = self.env.reset()
        obs = self._render()
        return obs

    def step(self, action):
        _, rew, done, info = self.env.step(action)
        next_obs = self._render()
        return next_obs, rew, done, info

    def _render(self):
        obs = self.env.render(resolution=(self.height, self.width),
                              offscreen=True,
                              camera_name=self.camera_name)
        return obs.copy()  # [h,w,n_channels]

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.env.close()


class FrameStackWrapper:
    def __init__(self, env, n_frames):
        self.env = env
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)

        wrapped_obs_shape = env.observation_space.shape
        
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=np.concatenate(
                [wrapped_obs_shape[:-1], [wrapped_obs_shape[-1] * n_frames]],
                axis=0),
            dtype=np.uint8)

        self.action_space = env.action_space

    @property
    def metadata(self):
        return self.env.metadata

    @property
    def obs_space(self):
        spaces = copy.deepcopy(self.env.obs_space)
        spaces['image'] = self.observation_space
        return spaces

    @property
    def act_space(self):
        return {'action': self.env.action_space}

    def reset(self):
        data = self.env.reset()
        obs = data[0]
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return (self._transform_observation(self.frames), *data[1:])

    def step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        self.frames.append(next_obs)
        return self._transform_observation(self.frames), rew, done, info

    def _transform_observation(self, obs):
        obs = np.concatenate(list(obs), axis=-1)
        return obs

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.env.close()


class FullStateWrapper():
    def __init__(self, env):
        self.env = env
        self.observation_space = spaces.Box(
            low=float('-inf'),
            high=float('inf'),
            shape=(31,),
            dtype=np.float32)

        self.action_space = env.action_space

    def reset(self):
        obs = self.env.reset()
        full_state = self.env.get_env_state()[0]
        print(full_state)
        full_state = np.concatenate([full_state.qpos,full_state.qvel])
        return full_state

    def step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        full_state = self.env.get_env_state()[0]
        full_state = np.concatenate([full_state.qpos,full_state.qvel])
        return full_state, rew, done, info

    def close(self):
        self.env.close()

    def seed(self,seed):
        self.env.seed(seed)


def make_metaworld(name="ML45",
                   obs_type='pixels',
                   n_frames=16,
                   height=96,
                   width=96,
                   n_channels=3,
                   camera_name='corner3'):
    assert name in ['ML10', 'ML45', 'MT10', 'ML50']
    assert obs_type in ['pixels', 'full_state']

    if name == 'ML10':
        ml = metaworld.ML10()
    elif name == 'ML45':
        ml = metaworld.ML45()
    elif name == 'MT10':
        ml = metaworld.MT10()
    elif name == 'ML50':
        ml = metaworld.MT50()
    else:
        raise NotImplementedError()

    training_envs = []
    
    for name, env_cls in ml.train_classes.items():
        env = env_cls()
        task = random.choice(
            [task for task in ml.train_tasks if task.env_name == name])
        env.set_task(task)
        if obs_type == 'pixels':
            env = PixelWrapper(env,
                               height=height,
                               width=width,
                               n_channels=n_channels,
                               camera_name=camera_name)
        # elif obs_type == 'full_state':
        #     env = FullStateWrapper(env)
        env = FrameStackWrapper(env, n_frames=n_frames) if n_frames > 1 else env
        training_envs.append(env)

    testing_envs = []
    for name, env_cls in ml.test_classes.items():
        env = env_cls()
        task = random.choice(
            [task for task in ml.test_tasks if task.env_name == name])
        env.set_task(task)
        if obs_type == 'pixel':
            env = PixelWrapper(env,
                               height=height,
                               width=width,
                               n_channels=n_channels,
                               camera_name=camera_name)
            env = FrameStackWrapper(env, n_frames=n_frames) if n_frames > 1 else env
        testing_envs.append(env)

    return training_envs, testing_envs
