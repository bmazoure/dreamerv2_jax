import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from jax.random import PRNGKey

tfd = tfp.distributions
tfb = tfp.bijectors



class SampleDist(tfd.Distribution):
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def _parameter_properties(self, *args, **kwargs):
        return self._dist._parameter_properties(*args, **kwargs)

    def sample(self, seed):
        return self._dist.sample(seed=seed)

    def mean(self, key):
        samples = self._dist.sample(self._samples, seed=key)
        return samples.mean(0)

    def mode(self, key):
        sample = self._dist.sample(self._samples, seed=key)
        logprob = self._dist.log_prob(sample)
        return jnp.take_along_axis(sample,jnp.expand_dims(jnp.stack([logprob.argmax(0)]*sample.shape[-1],-1),0),0)[0]

    def entropy(self, key):
        sample = self._dist.sample(self._samples, seed=key)
        logprob = self.log_prob(sample)
        return -logprob.mean(0)


class OneHotDist(tfd.OneHotCategorical):

    def mode(self):
        return super().mode().astype(self._sample_dtype)

    def sample(self, sample_shape=(), seed=None):
        # Straight through biased gradient estimator.
        sample = super().sample(sample_shape, seed)
        probs = self._pad(super().probs_parameter(), sample.shape)
        sample += (probs - jax.lax.stop_gradient(probs))
        return sample

    def _pad(self, tensor, shape):
        tensor = super().probs_parameter()
        while len(tensor.shape) < len(shape):
            tensor = tensor[None]
        return tensor


class TruncNormalDist(tfd.TruncatedNormal):
    def __init__(self, *args, clip=1e-6, mult=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._clip = clip
        self._mult = mult

    def sample(self, *args, **kwargs):
        event = super().sample(*args, **kwargs)
        if self._clip:
            clipped = jnp.clip(event, a_min=self.low + self._clip,
                                      a_max=self.high - self._clip)
            event = event - jax.lax.stop_gradient(event) + jax.lax.stop_gradient(clipped)
        if self._mult:
            event *= self._mult
        return event

# class TanhNormalDist():
#     def __init__(self, dist):
#         self.dist = dist
    
#     def entropy(self, *args, **kwargs):
#         import ipdb;ipdb.set_trace()
#         return self.dist.entropy(*args, **kwargs)
    
#     def mode(self, *args, **kwargs):
#         return self.dist.mode(*args, **kwargs)

#     @property
#     def name(self):
#         return "tanh_normal"

#     @property
#     def dtype(self):
#         return jnp.float32