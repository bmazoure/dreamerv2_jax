import numpy as np
import copy


class Driver:
    def __init__(self, envs, num_envs, init_state, **kwargs):
        self._envs = envs
        self.num_envs = num_envs
        self._kwargs = kwargs
        self._on_steps = []
        self._on_resets = []
        self._on_episodes = []
        self._act_spaces = envs.act_space
        self._state = init_state
        self.reset()
        self.infos = [[] for _ in range(self.num_envs)]
        self.steps = 0

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_reset(self, callback):
        self._on_resets.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def reformat_obs(self, obs, reward, done, info):
        [info[i].update(image=obs[i]) for i in range(obs.shape[0])]
        keys = set(info[0].keys()).intersection(*info[1:])

        joined_info = {k: np.stack([inf[k] for inf in info], 0) for k in keys}
        return joined_info, info

    def reset(self):
        obs = self._envs.reset()
        self._obs, split_obs = self.reformat_obs(*obs)
        action = np.zeros(self._act_spaces["action"].shape)
        [ob.update(action=action) for ob in split_obs]
        self.infos = [[] for _ in range(self.num_envs)]
        self._eps = [[ob] for ob in split_obs]
        self.state_needs_reset = True

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            actions, self._state = policy(self._obs, self._state, reset=self.state_needs_reset, **self._kwargs)
            self.state_needs_reset = False
            if type(actions) == dict:
                actions = np.asarray(actions['action'])

            obs, rew, done, infos = self._envs.step(actions)
            [self.infos[i].append(info) for i, info in enumerate(infos)]

            self._obs, split_obs = self.reformat_obs(obs, rew, done, infos)

            [fn(self._obs) for fn in self._on_steps]
            episode += self._obs["is_last"].sum()
            step += self.num_envs - self._obs["is_last"].sum()

            if self._obs["is_last"].sum() > 0:
                done_eps = [i for i, done in enumerate(self._obs["is_last"]) if done]
                try:
                    [fn(self.infos[i]) for fn in self._on_episodes for i in done_eps]
                except Exception as e:
                    print(e)
                [self.infos[i].clear() for i in done_eps]

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint64):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        return value
