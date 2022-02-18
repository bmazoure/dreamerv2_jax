import collections
import contextlib
import re
import time

from functools import partial

import jax.numpy as jnp
import flax.linen as nn

import numpy as np

from . import dists
from . import tfutils

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from jax.random import PRNGKey
import jax


class RandomAgent:
    def __init__(self, act_space, seed, logprob=False):
        self.act_space = act_space['action']
        self.logprob = logprob
        self.rng = PRNGKey(seed)
        if hasattr(self.act_space, 'n'):
            self._dist = dists.OneHotDist(jnp.zeros(self.act_space.n))
        else:
            dist = tfd.Uniform(self.act_space.low, self.act_space.high)
            self._dist = tfd.Independent(dist, 1)

    def __call__(self, obs, state=None, mode=None, **kwargs):
        self.rng, key = jax.random.split(self.rng)
        action = self._dist.sample(len(obs['is_first']), seed=key,)
        output = {'action': action}
        if self.logprob:
            output['logprob'] = self._dist.log_prob(action)
        return output, None


def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        step = tf.cast(step, tf.float32)
        match = re.match(r'linear\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [
                float(group) for group in match.groups()
            ]
            mix = tf.clip_by_value(step / duration, 0, 1)
            return (1 - mix) * initial + mix * final
        match = re.match(r'warmup\((.+),(.+)\)', string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = tf.clip_by_value(step / warmup, 0, 1)
            return scale * value
        match = re.match(r'exp\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, halflife = [
                float(group) for group in match.groups()
            ]
            return (initial - final) * 0.5**(step / halflife) + final
        match = re.match(r'horizon\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [
                float(group) for group in match.groups()
            ]
            mix = tf.clip_by_value(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)


def action_noise(action, amount, discrete):
    if amount == 0:
        return action
    if discrete:
        probs = amount / action.shape[-1] + (1 - amount) * action
        return dists.OneHotDist(probs=probs).sample()
    else:
        return jnp.clip(tfd.Normal(action, amount).sample(), -1, 1)


class RollingNorm():
    def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8):
        # Momentum of 0 normalizes only based on the current batch.
        # Momentum of 1 disables normalization.
        self._shape = tuple(shape)
        self._momentum = momentum
        self._scale = scale
        self._eps = eps
        self.reset()

    def __call__(self, inputs):
        self.update(inputs)
        outputs = self.transform(inputs)
        return outputs

    def reset(self):
        self.mag = jnp.ones((1,))

    def update(self, inputs):
        mag = jnp.abs(inputs.mean())
        self.mag = self._momentum * self.mag + (1 - self._momentum) * mag

    def transform(self, inputs):
        values = inputs
        values /= self.mag.astype(inputs.dtype) + self._eps
        values *= self._scale
        return values


class Timer:
    def __init__(self):
        self._indurs = collections.defaultdict(list)
        self._outdurs = collections.defaultdict(list)
        self._start_times = {}
        self._end_times = {}

    @contextlib.contextmanager
    def section(self, name):
        self.start(name)
        yield
        self.end(name)

    def wrap(self, function, name):
        def wrapped(*args, **kwargs):
            with self.section(name):
                return function(*args, **kwargs)

        return wrapped

    def start(self, name):
        now = time.time()
        self._start_times[name] = now
        if name in self._end_times:
            last = self._end_times[name]
            self._outdurs[name].append(now - last)

    def end(self, name):
        now = time.time()
        self._end_times[name] = now
        self._indurs[name].append(now - self._start_times[name])

    def result(self):
        metrics = {}
        for key in self._indurs:
            indurs = self._indurs[key]
            outdurs = self._outdurs[key]
            metrics[f'timer_count_{key}'] = len(indurs)
            metrics[f'timer_inside_{key}'] = np.sum(indurs)
            metrics[f'timer_outside_{key}'] = np.sum(outdurs)
            indurs.clear()
            outdurs.clear()
        return metrics


class CarryOverState:
    def __init__(self, fn, init_state=None):
        self._fn = fn
        self._state = init_state

    def __call__(self, *args):
        self._state, out, rec_obs = self._fn(*args, *self._state)
        return out, rec_obs

def schedule(string, step):
  try:
    return float(string)
  except ValueError:
    step = tf.cast(step, tf.float32)
    match = re.match(r'linear\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, duration = [float(group) for group in match.groups()]
      mix = tf.clip_by_value(step / duration, 0, 1)
      return (1 - mix) * initial + mix * final
    match = re.match(r'warmup\((.+),(.+)\)', string)
    if match:
      warmup, value = [float(group) for group in match.groups()]
      scale = tf.clip_by_value(step / warmup, 0, 1)
      return scale * value
    match = re.match(r'exp\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, halflife = [float(group) for group in match.groups()]
      return (initial - final) * 0.5 ** (step / halflife) + final
    match = re.match(r'horizon\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, duration = [float(group) for group in match.groups()]
      mix = tf.clip_by_value(step / duration, 0, 1)
      horizon = (1 - mix) * initial + mix * final
      return 1 - 1 / horizon
    raise NotImplementedError(string)