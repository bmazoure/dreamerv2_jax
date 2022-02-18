import atexit
import os
import sys
import threading
import traceback

import cloudpickle
import gym
import numpy as np
import multiprocessing as mp
from enum import Enum
from copy import deepcopy

from gym.error import (
    AlreadyPendingCallError,
    NoAsyncCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
)
from gym.vector.utils import (
    create_shared_memory,
    create_empty_array,
    write_to_shared_memory,
    read_from_shared_memory,
    concatenate,
    CloudpickleWrapper,
    clear_mpi_env_vars,
)

from gym.vector.async_vector_env import AsyncState


class GymWrapper:
    def __init__(self, env, obs_key='image', act_key='action'):
        self._env = env
        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
        self._act_is_dict = hasattr(self._env.action_space, 'spaces')
        self._obs_key = obs_key
        self._act_key = act_key

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return {
            **spaces,
            'reward':
            gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first':
            gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last':
            gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal':
            gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, done, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs['reward'] = float(reward)
        obs['is_first'] = False
        obs['is_last'] = done
        obs['is_terminal'] = info.get('is_terminal', done)
        return obs

    def reset(self):
        obs = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs['reward'] = 0.0
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False
        return obs


class DMC:
    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
        os.environ['MUJOCO_GL'] = 'egl'
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if domain == 'manip':
            from dm_control import manipulation
            self._env = manipulation.load(task + '_vision')
        elif domain == 'locom':
            from dm_control.locomotion.examples import basic_rodent_2020
            self._env = getattr(basic_rodent_2020, task)()
        else:
            from dm_control import suite
            self._env = suite.load(domain, task)
        self._action_repeat = action_repeat
        self._size = size
        if camera in (-1, None):
            camera = dict(
                quadruped_walk=2,
                quadruped_run=2,
                quadruped_escape=2,
                quadruped_fetch=2,
                locom_rodent_maze_forage=1,
                locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera
        self._ignored_keys = []
        for key, value in self._env.observation_spec().items():
            if value.shape == (0, ):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

    @property
    def metadata(self):
        return None

    @property
    def obs_space(self):
        spaces = {
            'image': gym.spaces.Box(0, 255, self._size + (3, ),
                                    dtype=np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }
        for key, value in self._env.observation_spec().items():
            if key in self._ignored_keys:
                continue
            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape,
                                             np.float32)
            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
            else:
                raise NotImplementedError(value.dtype)
        return spaces

    @property
    def act_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return {'action': action}

    @property
    def observation_space(self):
        return self.obs_space["image"]

    @property
    def action_space(self):
        return self.act_space["action"]

    def seed(self, seed):
        pass

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break
        assert time_step.discount in (0, 1)
        image = self._env.physics.render(*self._size, camera_id=self._camera)
        obs = {
            'reward': reward,
            'is_first': False,
            'is_last': time_step.last(),
            'is_terminal': time_step.discount == 0,
        }
        obs.update({
            k: v
            for k, v in dict(time_step.observation).items()
            if k not in self._ignored_keys
        })
        return image, reward, time_step.discount == 0, obs

    def reset(self):
        time_step = self._env.reset()
        image = self._env.physics.render(*self._size,
                                         camera_id=self._camera)
        obs = {
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
            'action': (self.action_space.high + self.action_space.low) / 2
        }
        obs.update({
            k: v
            for k, v in dict(time_step.observation).items()
            if k not in self._ignored_keys
        })
        return image, 0, False, obs

    def close(self):
        return self._env.close()


class Atari:

    LOCK = threading.Lock()

    def __init__(self,
                 name,
                 action_repeat=4,
                 size=(84, 84),
                 grayscale=True,
                 noops=30,
                 life_done=False,
                 sticky=True,
                 all_actions=False):
        assert size[0] == size[1]
        import gym.wrappers
        import gym.envs.atari
        if name == 'james_bond':
            name = 'jamesbond'
        with self.LOCK:
            env = gym.envs.atari.AtariEnv(
                game=name,
                obs_type='image',
                frameskip=1,
                repeat_action_probability=0.25 if sticky else 0.0,
                full_action_space=all_actions)
        # Avoid unnecessary rendering in inner env.
        env._get_obs = lambda: None
        # Tell wrapper that the inner env has no action repeat.
        env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
        self._env = gym.wrappers.AtariPreprocessing(env, noops, action_repeat,
                                                    size[0], life_done,
                                                    grayscale)
        self._size = size
        self._grayscale = grayscale

    @property
    def obs_space(self):
        shape = self._size + (1 if self._grayscale else 3, )
        return {
            'image': gym.spaces.Box(0, 255, shape, np.uint8),
            'ram': gym.spaces.Box(0, 255, (128, ), np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    @property
    def act_space(self):
        return {'action': self._env.action_space}

    def step(self, action):
        image, reward, done, info = self._env.step(action['action'])
        if self._grayscale:
            image = image[..., None]
        return {
            'image': image,
            'ram': self._env.env._get_ram(),
            'reward': reward,
            'is_first': False,
            'is_last': done,
            'is_terminal': done,
        }

    def reset(self):
        with self.LOCK:
            image = self._env.reset()
        if self._grayscale:
            image = image[..., None]
        return {
            'image': image,
            'ram': self._env.env._get_ram(),
            'reward': 0.0,
            "action": np.zeros(self.act_space["action"].shape),
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }

    def close(self):
        return self._env.close()


class Crafter:
    def __init__(self, outdir=None, reward=True, seed=None):
        import crafter
        self._env = crafter.Env(reward=reward, seed=seed)
        self._env = crafter.Recorder(
            self._env,
            outdir,
            save_stats=True,
            save_video=False,
            save_episode=False,
        )
        self._achievements = crafter.constants.achievements.copy()

    @property
    def obs_space(self):
        spaces = {
            'image': self._env.observation_space,
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'log_reward': gym.spaces.Box(-np.inf, np.inf, (), np.float32),
        }
        spaces.update({
            f'log_achievement_{k}': gym.spaces.Box(0, 2**31 - 1, (), np.int32)
            for k in self._achievements
        })
        return spaces

    @property
    def act_space(self):
        return {'action': self._env.action_space}

    def step(self, action):
        image, reward, done, info = self._env.step(action['action'])
        obs = {
            'image': image,
            'reward': reward,
            'is_first': False,
            'is_last': done,
            'is_terminal': info['discount'] == 0,
            'log_reward': info['reward'],
        }
        obs.update({
            f'log_achievement_{k}': v
            for k, v in info['achievements'].items()
        })
        return obs

    def reset(self):
        obs = {
            'image': self._env.reset(),
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
            "action": np.zeros(self.act_space["action"].shape),
            'log_reward': 0.0,
        }
        obs.update({f'log_achievement_{k}': 0 for k in self._achievements})
        return obs, 0, False, obs


class MetaWorld:
    def __init__(self, name="ML45", type_='train', framestack=1, dims=[96, 96], n_channels=3, camera='corner3', start_task=0, n_tasks=2, obs_type='pixels'):
        from .wrappers import make_metaworld, SequentialTaskWrapper
        train_envs, test_envs = make_metaworld(name=name,
                                               obs_type=obs_type,
                                               n_frames=framestack,
                                               height=dims[0],
                                               width=dims[1],
                                               n_channels=n_channels,
                                               camera_name=camera)
        if n_tasks > 0:
            train_envs = train_envs[start_task:start_task+n_tasks]
            # test_envs = test_envs[start_task:start_task+n_tasks]
        else:
            train_envs = train_envs[start_task:]
            # test_envs = test_envs[start_task:]
        if type_ == 'train':
            self._env = SequentialTaskWrapper(train_envs)
        else:
            self._env = SequentialTaskWrapper(test_envs)
        
        # seed = np.random.randint(0,10000,1).item()
        # for image_id in range(5):
        #     grid = np.zeros((4*96, 3*96, 3), dtype=np.uint8)
        #     r, c = 0, 0
        #     for i in range(len(train_envs)):
        #         x = train_envs[i].reset().copy()
        #         grid[r*96:(r+1)*96, c*96:(c+1)*96] = x
        #         c += 1
        #         if c > 2:
        #             r += 1
        #             c = 0
            
        #     import matplotlib.pyplot as plt
        #     plt.imshow(grid)
        #     plt.savefig(name+'_'+str(seed + image_id))
        #     plt.clf()
        # exit()

    @property
    def obs_space(self):
        spaces = {
            'image': self._env.observation_space,
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'log_reward': gym.spaces.Box(-np.inf, np.inf, (), np.float32),
            'task': gym.spaces.Box(0, 50, (), np.int32),
            'log_success': gym.spaces.Box(0, 1, (), dtype=np.bool)
        }
        return spaces

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def act_space(self):
        return {'action': self._env.action_space}

    @property
    def action_space(self):
        return self._env.action_space

    def seed(self, seed):
        self._env.seed(seed)

    def step(self, action):
        image, reward, done, info = self._env.step(action)
        info = {
            'reward': reward,
            'is_first': False,
            'is_last': done,
            'is_terminal': done,
            'log_reward': reward,
            "action": action,
            'log_task': self._env.task_id,
            "log_{}_success".format(self._env.task_id): info["success"],
            "log_success".format(self._env.task_id): info["success"],
            **info
        }
        return image, reward, done, info

    def reset(self):
        image = self._env.reset()
        info = {
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
            "action": np.zeros(self.action_space.shape),
            'log_reward': 0.0,
            'log_task': self._env.task_id,
            "log_{}_success".format(self._env.task_id): 0,
            "log_success": 0,
        }
        return image, 0, False, info

    def close(self):
        return self._env.close()


class Dummy:
    def __init__(self):
        pass

    @property
    def obs_space(self):
        return {
            'image': gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    @property
    def act_space(self):
        return {'action': gym.spaces.Box(-1, 1, (6, ), dtype=np.float32)}

    def step(self, action):
        return {
            'image': np.zeros((64, 64, 3)),
            'reward': 0.0,
            'is_first': False,
            'is_last': False,
            'is_terminal': False,
        }

    def reset(self):
        return {
            'image': np.zeros((64, 64, 3)),
            'reward': 0.0,
            "action": np.zeros(self.act_space["action"].shape),
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }


class DelayedResetWrapper:
    def __init__(self, env):
        self.env = env
        self.done = True

    def step(self, action):
        if self.done:
            self.done = False
            return self.env.reset()

        obs, reward, done, info = self.env.step(action)
        if info["is_last"] or done:
            self.done = True

        return obs, reward, done, info

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self.env, name)
        except AttributeError:
            raise ValueError(name)


class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, rew, done, info = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            info['is_last'] = True
            done = True
            self._step = None
        return obs, rew, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class NormalizeAction:
    def __init__(self, env, key='action'):
        self._env = env
        self._key = key
        space = env.action_space
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        if self._key is None:
            orig = (action + 1) / 2 * (self._high - self._low) + self._low
            orig = np.where(self._mask, orig, action)
            result = self._env.step(orig)
            result[-1]["action"] = action
            return result

        else:
            orig = (action[self._key] + 1) / 2 * (self._high -
                                                  self._low) + self._low
            orig = np.where(self._mask, orig, action[self._key])
            result = self._env.step({**action, self._key: orig})
            result[-1]["action"] = action
            return result

class OneHotAction:
    def __init__(self, env, key='action'):
        assert hasattr(env.act_space[key], 'n')
        self._env = env
        self._key = key
        self._random = np.random.RandomState()

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        shape = (self._env.act_space[self._key].n, )
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.n = shape[0]
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        index = np.argmax(action[self._key]).astype(int)
        reference = np.zeros_like(action[self._key])
        reference[index] = 1
        if not np.allclose(reference, action[self._key]):
            raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step({**action, self._key: index})

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.act_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class ResizeImage:
    def __init__(self, env, size=(64, 64)):
        self._env = env
        self._size = size
        self._keys = [
            k for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size
        ]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image
            self._Image = Image

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def reset(self):
        obs = self._env.reset()
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


class RenderImage:
    def __init__(self, env, key='image'):
        self._env = env
        self._key = key
        self._shape = self._env.render().shape

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        obs[self._key] = self._env.render('rgb_array')
        return obs

    def reset(self):
        obs = self._env.reset()
        obs[self._key] = self._env.render('rgb_array')
        return obs


class Async:

    # Message types for communication via the pipe.
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _CLOSE = 4
    _EXCEPTION = 5

    def __init__(self, constructor, strategy='thread'):
        self._pickled_ctor = cloudpickle.dumps(constructor)
        if strategy == 'process':
            import multiprocessing as mp
            context = mp.get_context('spawn')
        elif strategy == 'thread':
            import multiprocessing.dummy as context
        else:
            raise NotImplementedError(strategy)
        self._strategy = strategy
        self._conn, conn = context.Pipe()
        self._process = context.Process(target=self._worker, args=(conn, ))
        atexit.register(self.close)
        self._process.start()
        self._receive()  # Ready.
        self._obs_space = None
        self._act_space = None

    def access(self, name):
        self._conn.send((self._ACCESS, name))
        return self._receive

    def call(self, name, *args, **kwargs):
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            pass  # The connection was already closed.
        self._process.join(5)

    @property
    def obs_space(self):
        if not self._obs_space:
            self._obs_space = self.access('obs_space')()
        return self._obs_space

    @property
    def act_space(self):
        if not self._act_space:
            self._act_space = self.access('act_space')()
        return self._act_space

    def step(self, action, blocking=False):
        promise = self.call('step', action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=False):
        promise = self.call('reset')
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        try:
            message, payload = self._conn.recv()
        except (OSError, EOFError):
            raise RuntimeError('Lost connection to environment worker.')
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError(
            'Received message of unexpected type {}'.format(message))

    def _worker(self, conn):
        try:
            ctor = cloudpickle.loads(self._pickled_ctor)
            env = ctor()
            conn.send((self._RESULT, None))  # Ready.
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    break
                raise KeyError(
                    'Received message of unknown type {}'.format(message))
        except Exception:
            stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
            print('Error in environment process: {}'.format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))
        finally:
            try:
                conn.close()
            except IOError:
                pass  # The connection was already closed.


def _worker_shared_memory_info_reset(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation, reward, done, info = env.reset()
                write_to_shared_memory(
                    index, observation, shared_memory, observation_space
                )
                pipe.send(((None, reward, done, info), True))
            elif command == "step":
                observation, reward, done, info = env.step(data)
                # if done:
                #     observation = env.reset()
                write_to_shared_memory(
                    index, observation, shared_memory, observation_space
                )
                pipe.send(((None, reward, done, info), True))
            elif command == "seed":
                env.seed(data)
                np.random.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_check_observation_space":
                pipe.send((data == observation_space, True))
            else:
                raise RuntimeError(
                    "Received unknown command `{0}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, "
                    "`_check_observation_space`}.".format(command)
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


class InfoResetAsyncVectorEnv(gym.vector.AsyncVectorEnv):

    def __init__(
        self,
        env_fns,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
    ):
        ctx = mp.get_context(context)
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy
        dummy_env = env_fns[0]()

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or dummy_env.observation_space
            action_space = action_space or dummy_env.action_space

        self.obs_space = dummy_env.obs_space
        self.act_space = dummy_env.act_space

        dummy_env.close()
        del dummy_env

        super(gym.vector.AsyncVectorEnv, self).__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

        if self.shared_memory:
            try:
                _obs_buffer = create_shared_memory(
                    self.single_observation_space, n=self.num_envs, ctx=ctx
                )
                self.observations = read_from_shared_memory(
                    _obs_buffer, self.single_observation_space, n=self.num_envs
                )
            except CustomSpaceError:
                raise ValueError(
                    "Using `shared_memory=True` in `AsyncVectorEnv` "
                    "is incompatible with non-standard Gym observation spaces "
                    "(i.e. custom spaces inheriting from `gym.Space`), and is "
                    "only compatible with default Gym spaces (e.g. `Box`, "
                    "`Tuple`, `Dict`) for batching. Set `shared_memory=False` "
                    "if you use custom observation spaces."
                )
        else:
            _obs_buffer = None
            self.observations = create_empty_array(
                self.single_observation_space, n=self.num_envs, fn=np.zeros
            )

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = worker or _worker_shared_memory_info_reset
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name="Worker<{0}>-{1}".format(type(self).__name__, idx),
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        _obs_buffer,
                        self.error_queue,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT
        # self._check_observation_spaces()

    def reset_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `reset_wait` times out. If
            `None`, the call to `reset_wait` never times out.
        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_wait` without any prior " "call to `reset_async`.",
                AsyncState.WAITING_RESET.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                "The call to `reset_wait` has timed out after "
                "{0} second{1}.".format(timeout, "s" if timeout > 1 else "")
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT
        observations_list, rewards, dones, infos = zip(*results)

        if not self.shared_memory:
            self.observations = concatenate(
                observations_list, self.observations, self.single_observation_space
            )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.array(rewards),
            np.array(dones, dtype=np.bool_),
            infos,
        )

    def reset_subset(self, indices, timeout=None):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `reset_wait` times out. If
            `None`, the call to `reset_wait` never times out.
        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `reset_async` while waiting "
                "for a pending call to `{0}` to complete".format(self._state.value),
                self._state.value,
            )

        for i in indices:
            self.parent_pipes[i].send(("reset", None))
        self._state = AsyncState.WAITING_RESET

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                "The call to `reset_wait` has timed out after "
                "{0} second{1}.".format(timeout, "s" if timeout > 1 else "")
            )

        results, successes = zip(*[self.parent_pipes[i].recv() for i in indices])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT
        observations_list, rewards, dones, infos = zip(*results)

        if not self.shared_memory:
            self.observations = concatenate(
                observations_list, self.observations, self.single_observation_space
            )

        observations = self.observations[indices]
        return (
            deepcopy(observations)if self.copy else observations,
            np.array(rewards),
            np.array(dones, dtype=np.bool_),
            infos,
        )


class InfoResetSyncVectorEnv(gym.vector.SyncVectorEnv):
    """Vectorized environment that serially runs multiple environments.
    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.
    observation_space : `gym.spaces.Space` instance, optional
        Observation space of a single environment. If `None`, then the
        observation space of the first environment is taken.
    action_space : `gym.spaces.Space` instance, optional
        Action space of a single environment. If `None`, then the action space
        of the first environment is taken.
    copy : bool (default: `True`)
        If `True`, then the `reset` and `step` methods return a copy of the
        observations.
    """

    def reset_wait(self):
        self._dones[:] = False
        observations, rewards, dones, infos = zip(*[e.reset() for e in self.envs])
        self.observations = concatenate(
            observations, self.observations, self.single_observation_space
        )
        self._rewards = np.stack(rewards, 0)
        self._dones = np.stack(dones, 0)
        return (deepcopy(self.observations) if self.copy else self.observations,
                np.copy(self._rewards),
                np.copy(self._dones), infos)

    def reset_subset(self, indices):
        self._dones[:] = False
        observations, rewards, dones, infos = zip([self.envs[env_id].reset() for env_id in indices])
        for i, env in enumerate(indices):
            self.observations[env] = observations[i]
            self._rewards[env] = observations[i]
            self._dones[env] = dones[i]

        return (deepcopy(self.observations) if self.copy else self.observations,
                np.copy(self._rewards),
                np.copy(self._dones), infos)

    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            if isinstance(observation, tuple):
                observation = observation[0]
            observations.append(observation)
            infos.append(info)
        self.observations = concatenate(
            observations, self.observations, self.single_observation_space
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._dones),
            infos,
        )

    @property
    def obs_space(self):
        return self.envs[0].obs_space

    @property
    def act_space(self):
        return self.envs[0].act_space