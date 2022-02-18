import re
from typing import Any, Optional, Tuple, Sequence, Callable, List
from functools import partial

import jax.numpy as jnp
import jax
import flax.linen as nn

import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from common.dists import OneHotDist
from jax.random import PRNGKey
from flax.core import FrozenDict

import common


def default_conv_init():
    # return nn.initializers.xavier_uniform() # Jax
    return nn.initializers.glorot_uniform()  # Tf init


def default_mlp_init():
    # return nn.initializers.orthogonal(0.01)
    return nn.initializers.glorot_uniform()  # Tf init


def default_gru_kernel_init():
    return nn.initializers.glorot_uniform()  # Tf init


@jax.jit
def mask(tensor, mask):
    return jnp.einsum('b,b...->b...', mask, tensor)


def swap(x):
    return jnp.transpose(x, [1, 0] + list(range(2, len(x.shape))))


def relaxed_kl(a, b):
    a_logits = a.distribution.logits
    b_logits = b.distribution.logits
    return (jax.nn.softmax(a_logits) * (jax.nn.log_softmax(a_logits) - jax.nn.log_softmax(b_logits))).sum(-1).sum(-1)


class EnsembleRSSM(nn.Module):
    ensemble: int = 5
    stoch: int = 30
    gru_hidden: int = 200
    gru_layers: int = 1
    mlp_hidden: int = 200
    discrete: int = 32
    act: str = 'elu'
    norm: bool = False
    std_act: str = 'softplus'
    min_std: float = 0.1
    batch: int = 16
    dtype: jnp.dtype = jnp.float32
    seed: int = 123
    config_kl: FrozenDict = None

    def setup(self):
        self.lin1 = nn.Dense(self.mlp_hidden,
                             kernel_init=default_mlp_init(),
                             dtype=self.dtype)
        self.ln1 = nn.LayerNorm(dtype=self.dtype)
        self.lin2 = nn.Dense(self.mlp_hidden,
                             kernel_init=default_mlp_init(),
                             dtype=self.dtype)
        self.ln2 = nn.LayerNorm(dtype=self.dtype)
        self.lin3 = nn.Dense(self.stoch * self.discrete,
                             kernel_init=default_mlp_init(),
                             dtype=self.dtype)
        self.lin1_ensemble = [nn.Dense(self.mlp_hidden,
                                       kernel_init=default_mlp_init(),
                                       dtype=self.dtype) for _ in range(self.ensemble)]
        self.ln1_ensemble = [nn.LayerNorm(dtype=self.dtype) for _ in range(self.ensemble)]
        if self.discrete:
            self.lin2_ensemble = [nn.Dense(self.stoch * self.discrete,
                                           kernel_init=default_mlp_init(),
                                           dtype=self.dtype) for _ in range(self.ensemble)]
            self.lin4 = nn.Dense(self.stoch * self.discrete, dtype=self.dtype)
        else:
            self.lin2_ensemble = [nn.Dense(2 * self.stoch, kernel_init=default_mlp_init(), dtype=self.dtype) for _ in
                                  range(self.ensemble)]
            self.lin4 = nn.Dense(2 * self.stoch, kernel_init=default_mlp_init(), dtype=self.dtype)
        self.gru_cell = [nn.GRUCell(activation_fn=nn.tanh,
                                    kernel_init=default_gru_kernel_init(),
                                    dtype=self.dtype
                                    )
                         for _ in range(self.gru_layers)]

        self.activation = get_activation(self.act)
        self.std_activation = get_activation(self.act)

        self.init_rssm(PRNGKey(self.seed))

    @nn.compact
    def __call__(self, embed, action, is_first, rng, imagine=False, batch_size=None):
        stats, stoch, det = self.init_rssm(rng, batch_size)
        key, rng = jax.random.split(rng)
        (stats_post, stoch_post, deter), (stats_prior, stoch_prior, deter) = self.observe(embed, action, is_first, key,
                                                                                          stats, stoch, det)
        kl_loss, kl_value = self.kl_loss((stats_post, stoch_post, deter), (stats_prior, stoch_prior, deter))
        key, rng = jax.random.split(rng)
        stats_prior, stoch_prior, det_prior = self.imagine(action, key, stats, stoch, det)
        return ((stats_post, stoch_post, deter), (stats_prior, stoch_prior, det_prior)), (kl_loss, kl_value)

    def init_rssm(self, rng, batch_size=None):
        if batch_size is None:
            batch_size = self.batch
        key, _ = jax.random.split(rng)
        gru_states = []
        for i, gru_cell in enumerate(self.gru_cell):
            gru_state = gru_cell.initialize_carry(rng,
                                                  batch_dims=(batch_size,),
                                                  size=self.gru_hidden).astype(self.dtype)
            gru_states.append(gru_state)
        gru_state = jnp.concatenate(gru_states, -1)

        if self.discrete:
            state = (jax.random.uniform(key, shape=(batch_size, self.stoch, self.discrete),
                                        dtype=self.dtype),
                     jax.random.uniform(key, shape=(batch_size, self.stoch, self.discrete),
                                        dtype=jnp.float32),
                     gru_state)
        else:
            state = ((jnp.zeros(shape=(batch_size, self.stoch),
                                dtype=self.dtype),
                      jnp.zeros(shape=(batch_size, self.stoch),
                                dtype=self.dtype)),
                     jnp.zeros(shape=(batch_size, self.stoch),
                               dtype=self.dtype),
                     gru_states)
        return state

    def img_step(self, stats, stoch, deter, prev_action, key, sample=True):
        prev_stoch = stoch.astype(self.dtype)
        prev_action = prev_action.astype(self.dtype)
        if self.discrete:
            shape = prev_stoch.shape[:-2] + (self.stoch * self.discrete,)
            prev_stoch = prev_stoch.reshape(shape)
        x = jnp.concatenate([prev_stoch, prev_action], -1)
        x = self.lin1(x)
        if self.norm:
            x = self.ln1(x)
        x = self.activation(x)

        new_deter = []
        for i, cell in enumerate(self.gru_cell):
            det = deter[..., i*self.gru_hidden:(i+1)*self.gru_hidden]
            x, new_deter_i = cell(x, det)
            new_deter.append(new_deter_i)

        deter = jnp.concatenate(new_deter, -1)
        stats = self._suff_stats_layer(0, x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=key) if sample else dist.mode()
        return stats, stoch, deter

    @partial(
        nn.transforms.scan,
        variable_broadcast='params',
        split_rngs={'params': False})
    def obs_step_scan(self, state, input):
        (prev_stats, prev_stoch, prev_det) = state
        prev_action, embed, is_first, rng = input
        prev_stoch = jnp.einsum('b,b...->b...', 1.0 - is_first, prev_stoch)
        prev_stats = jnp.einsum('b,b...->b...', 1.0 - is_first, prev_stats)
        prev_det = jnp.einsum('b,b...->b...', 1.0 - is_first, prev_det)

        prev_action = jnp.einsum('b,b...->b...', 1.0 - is_first, prev_action)
        rng, key = jax.random.split(rng)
        stats_prior, stoch_prior, deter = self.img_step(prev_stats, prev_stoch,
                                                        prev_det, prev_action,
                                                        key, True)
        x = jnp.concatenate([deter, embed], -1)
        x = self.lin2(x)
        if self.norm:
            x = self.ln2(x)
        x = self.activation(x)
        stats_post = self._suff_stats_layer(-1, x)
        dist = self.get_dist(stats_post)
        rng, key = jax.random.split(rng)
        stoch_post = dist.sample(seed=key)
        return (stats_post, stoch_post, deter), (stats_prior, stoch_prior,
                                                 stats_post, stoch_post, deter)

    def obs_step(self, prev_stats, prev_stoch, prev_det, prev_action, embed, is_first, rng, sample=True):
        prev_stoch = jnp.einsum('b,b...->b...', 1.0 - is_first, prev_stoch)
        prev_stats = jnp.einsum('b,b...->b...', 1.0 - is_first, prev_stats)
        prev_det = jnp.einsum('b,b...->b...', 1.0 - is_first, prev_det)

        prev_action = jnp.einsum('b,b...->b...', 1.0 - is_first, prev_action)

        rng, key = jax.random.split(rng)
        stats_prior, stoch_prior, deter = self.img_step(prev_stats, prev_stoch, prev_det, prev_action, key, sample)
        x = jnp.concatenate([deter, embed], -1)
        x = self.lin2(x)
        if self.norm:
            x = self.ln2(x)
        x = self.activation(x)
        stats_post = self._suff_stats_layer(-1, x)
        dist = self.get_dist(stats_post)
        rng, key = jax.random.split(rng)
        stoch_post = dist.sample(seed=key) if sample else dist.mode()
        return (stats_post, stoch_post, deter), (stats_prior, stoch_prior, deter)

    def observe(self, embed, action, is_first, rng, stats, stoch, det, reinit=False):
        rng, key = jax.random.split(rng)
        if reinit:
            stats, stoch, det = self.init_rssm(key)
        
        keys = []
        for i in jnp.arange(action.shape[1]):
            rng, key = jax.random.split(rng)
            keys.append(key)
        keys = jnp.array(keys)
        
        _, (stats_prior_acc,
            stoch_prior_acc,
            stats_post_acc,
            stoch_post_acc,
            deter_acc) = self.obs_step_scan((stats, stoch, det),
                                            (action.swapaxes(0, 1),
                                             embed.swapaxes(0, 1),
                                             is_first.swapaxes(0, 1), keys))

        return (stats_post_acc.swapaxes(1, 0),
                stoch_post_acc.swapaxes(1, 0),
                deter_acc.swapaxes(1, 0)), \
               (stats_prior_acc.swapaxes(1, 0),
                stoch_prior_acc.swapaxes(1, 0),
                deter_acc.swapaxes(1, 0))

    def imagine(self, action, rng, stats, stoch, det, reinit=False):
        rng, key = jax.random.split(rng)
        if reinit:
            stats, stoch, det = self.init_rssm(key)
        stoch_acc, deter_acc, stats_acc = [], [], []
        for i in jnp.arange(action.shape[1]):
            rng, key = jax.random.split(rng)
            stats, stoch, det = self.img_step(stats, stoch, det, action[:, i], key)
            stoch_acc.append(stoch)
            deter_acc.append(det)
            stats_acc.append(stats)
        stoch_acc = jnp.stack(stoch_acc, 0)
        deter_acc = jnp.stack(deter_acc, 0)
        stats_acc = jnp.stack(stats_acc, 0)
        return stoch_acc, deter_acc, stats_acc

    def get_feat(self, stoch, deter):
        if self.discrete:
            shape = stoch.shape[:-2] + (self.stoch * self.discrete,)
            stoch = stoch.reshape(shape)
        return jnp.concatenate([stoch, deter], -1)

    def get_dist(self, state, ensemble=False):
        if ensemble:
            state = self._suff_stats_ensemble(state['deter'])
        if self.discrete:
            logit = state
            logit = logit.astype(jnp.float32)
            if self.config_kl['discrete_temp'] > 0.:
                dist = tfd.Independent(
                    tfd.RelaxedOneHotCategorical(temperature=self.config_kl['discrete_temp'], logits=logit), 1)
            else:
                dist = tfd.Independent(common.OneHotDist(logits=logit, dtype=jnp.float32), 1)

        else:
            mean, std = state
            mean = mean.astype(jnp.float32)
            std = std.astype(jnp.float32)
            dist = tfd.MultivariateNormalDiag(mean, std)
        return dist

    def _suff_stats_ensemble(self, inp, index):
        bs = list(inp.shape[:-1])
        inp = inp.reshape([-1, inp.shape[-1]])
        if self.discrete:
            logits = []
        else:
            mean, std = [], []
        for k in range(self.ensemble):
            x = self.lin1_ensemble[k](inp)
            if not self.norm:
                x = self.ln1_ensemble[k](x)
            x = self.activation(x)
            stats = self._suff_stats_layer(k, x)
            if self.discrete:
                logits.append(stats)
            else:
                mean.append(stats[0])
                std.append(stats[1])
        if self.discrete:
            return logits[index]
        else:
            return mean[index], std[index]

    def _suff_stats_layer(self, k, x):
        if k > 0:
            net = self.lin2_ensemble[k]
        else:
            net = self.lin4
        if self.discrete:
            x = net(x)
            logit = jnp.reshape(x, x.shape[:-1] + (self.stoch, self.discrete))
            return logit
        else:
            x = net(x)
            mean, std = nn.split(x, 2, -1)
            std = self.std_activation(std) + self.min_std
            return mean, std

    def kl_loss(self, post, prior):
        kld = tfd.kl_divergence if self.config_kl['discrete_temp'] <= 0 else relaxed_kl
        lhs, rhs = (prior, post) if self.config_kl['forward'] else (post, prior)
        mix = self.config_kl['balance'] if self.config_kl['forward'] else (1 - self.config_kl['balance'])
        if self.config_kl['balance'] == 0.5:
            value = kld(self.get_dist(lhs[0]), self.get_dist(rhs[0]))
            loss = jnp.maximum(value, self.config_kl['free']).mean()
        else:
            value_lhs = value = kld(self.get_dist(lhs[0]), self.get_dist(jax.lax.stop_gradient(rhs[0])))
            value_rhs = kld(self.get_dist(jax.lax.stop_gradient(lhs[0])), self.get_dist(rhs[0]))
            if self.config_kl['free_avg']:
                loss_lhs = jnp.maximum(value_lhs.mean(), self.config_kl['free'])
                loss_rhs = jnp.maximum(value_rhs.mean(), self.config_kl['free'])
            else:
                loss_lhs = jnp.maximum(value_lhs, self.config_kl['free']).mean()
                loss_rhs = jnp.maximum(value_rhs, self.config_kl['free']).mean()
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss, value


class Encoder(nn.Module):
    shapes: Sequence[int]
    act: str = 'elu'
    norm: bool = False
    cnn_type: str = 'impala'
    mlp_layers: Sequence[int] = (512, 512, 512, 512, 256)
    mlp_keys: Sequence[str] = ()
    cnn_keys: Sequence[str] = ('image')
    dtype: Any = jnp.float32
    obs_type: str = 'pixels'

    def setup(self):
        if self.obs_type == 'full_state':
            self.obs_encoder = MLP(out_shape=self.mlp_layers[-1],
                                   hidden_dims=self.mlp_layers[:-1],
                                   act=self.act,
                                   activate_final=True,
                                   norm=self.norm,
                                   dist=None)
        elif self.obs_type == 'pixels':
            if self.cnn_type == 'impala':
                self.obs_encoder = Impala(prefix='encoder', dtype=self.dtype, )
            elif self.cnn_type == 'danijar':
                self.obs_encoder = DanijarEncoder(prefix='encoder', dtype=self.dtype, norm=self.norm, act=self.act,
                                                  cnn_depth=48)
        self.mlp_actions = MLP(out_shape=self.mlp_layers[-1],
                               hidden_dims=self.mlp_layers[:-1],
                               act=self.act,
                               activate_final=True,
                               norm=self.norm,
                               dist=None,
                               dtype=self.dtype)
        self.mlp_rewards = MLP(out_shape=self.mlp_layers[-1],
                               hidden_dims=self.mlp_layers[:-1],
                               act=self.act,
                               activate_final=True,
                               norm=self.norm,
                               dist=None,
                               dtype=self.dtype)

    def __call__(self, image, reward=None, action=None):
        batch_dims = image.shape[:2]
        img_dims = image.shape[2:]
        z = self.obs_encoder(image.reshape(-1, *img_dims))
        z = z.reshape(*batch_dims, -1)
        if action is not None:
            z_mlp_action = self.mlp_actions(action.reshape(-1, 1)).reshape(*batch_dims, -1)
            z = jnp.concatenate([z, z_mlp_action], -1)
        if reward is not None:
            z_mlp_reward = self.mlp_rewards(reward.reshape(-1, 1)).reshape(*batch_dims, -1)
            z = jnp.concatenate([z, z_mlp_reward], -1)
        # Now of shape n_batch x T x n_latent
        return z


class Decoder(nn.Module):
    shapes: Sequence[int]
    act: str = 'elu'
    norm: bool = False
    cnn_type: str = 'impala'
    mlp_layers: Sequence[int] = (512, 512, 512, 512)
    channels: int = 3
    framestack: int = 1
    mlp_keys: Sequence[str] = ()
    cnn_keys: Sequence[str] = ('image')
    dtype: Any = jnp.float32
    obs_type: str = 'pixels'

    def setup(self):
        if self.obs_type == 'full_state':
            self.obs_decoder = MLP(out_shape=(39,),
                                   hidden_dims=self.mlp_layers[:-1],
                                   act=self.act,
                                   activate_final=True,
                                   norm=self.norm,
                                   dtype=self.dtype,
                                   dist='mse')
            self.representation = nn.Dense(self.mlp_layers[-1],
                                           kernel_init=default_mlp_init(),
                                           dtype=self.dtype, )
        elif self.obs_type == 'pixels':
            if self.cnn_type == 'impala':
                if self.shapes['image'][:-1] == (64, 64):
                    n_repr = 2048
                elif self.shapes['image'][:-1] == (96, 96):
                    n_repr = 4608*2
                self.representation = nn.Dense(n_repr, kernel_init=default_mlp_init(),
                                               dtype=self.dtype, )
                self.obs_decoder = ImpalaDeconv(prefix='decoder',
                                                last_channel=self.channels * self.framestack,
                                                dtype=self.dtype, )
            elif self.cnn_type == 'danijar':
                self.representation = nn.Dense(32 * 48, kernel_init=default_mlp_init(),
                                               dtype=self.dtype)
                self.obs_decoder = DanijarDecoder(prefix='decoder',
                                                  last_channel=self.channels * self.framestack,
                                                  norm=self.norm, act=self.act, cnn_depth=48,
                                                  dtype=self.dtype)

    def __call__(self, features):
        # put states batch to n_batch x 2048
        features = features.reshape(-1, features.shape[-1])
        features = self.representation(features)
        if self.obs_type == 'full_state':
            dist = self.obs_decoder(features)
            # dist = tfd.Normal(x_hat, 1)
        elif self.obs_type == 'pixels':
            if self.cnn_type == 'impala':
                if self.shapes['image'][:-1] == (64, 64):
                    features = features.reshape(-1, 8, 8, 32)
                elif self.shapes['image'][:-1] == (96, 96):
                    features = features.reshape(-1, 12, 12, 64)
            elif self.cnn_type == 'danijar':
                features = features.reshape(-1, 1, 1, 32 * self.obs_decoder.cnn_depth)
            x_hat = self.obs_decoder(features)
            dist = tfd.Independent(tfd.Normal(0.5 * jnp.tanh(x_hat), 1), 3)
        return dist


class MLP(nn.Module):
    out_shape: Sequence[int]
    hidden_dims: Sequence[int]
    act: str = 'elu'
    activate_final: int = False
    norm: bool = False
    dist: str = 'mse'
    min_std: float = 0.1
    init_std: float = 0.0
    dtype: Any = jnp.float32

    def setup(self):
        self.activation = get_activation(self.act)

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_mlp_init(),
                         dtype=self.dtype)(x)
            if not self.norm:
                x = nn.LayerNorm()(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activation(x)
        if self.dist is not None:
            x = DistLayer(self.out_shape, self.dist, self.min_std,
                          self.init_std)(x)
        return x


class DistLayer(nn.Module):
    """ A simple layer that projects to the parameters of a distribution.

    This layer is always kept in float32, for numerical stability
    """
    shape: Sequence[int]
    dist: str = 'mse'
    min_std: float = 0.1
    init_std: float = 0.0
    # dtype: any = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        out = nn.Dense(np.prod(self.shape), kernel_init=default_mlp_init())(inputs)
        out = out.reshape(inputs.shape[:-1] + self.shape)
        if self.dist in ('normal', 'tanh_normal', 'trunc_normal'):
            std = nn.Dense(np.prod(self.shape), kernel_init=default_mlp_init())(inputs)
            std = std.reshape(inputs.shape[:-1] + self.shape)
        if self.dist == 'mse':
            dist = tfd.Normal(out, 1.0)
            return tfd.Independent(dist, len(self.shape))
        if self.dist == 'normal':
            dist = tfd.Normal(out, std)
            return tfd.Independent(dist, len(self.shape))
        if self.dist == 'binary':
            dist = tfd.Bernoulli(out)
            return tfd.Independent(dist, len(self.shape))
        if self.dist == 'tanh_normal':
            mean = 5 * jnp.tanh(out / 5)
            new_std = nn.softplus(std + self.init_std) + self.min_std
            dist = tfd.Normal(mean, new_std)
            dist = tfd.TransformedDistribution(distribution=dist,
                                               bijector=tfb.Chain([tfb.Tanh()
                                                                   ]))
            dist = tfd.Independent(dist, 1)
            dist = common.SampleDist(dist)
            return dist
        if self.dist == 'trunc_normal':
            std = 2 * nn.sigmoid((std + self.init_std) / 2) + self.min_std
            dist = common.TruncNormalDist(nn.tanh(out), std, -1, 1)
            return tfd.Independent(dist, 1)
        if self.dist == 'onehot':
            return OneHotDist(out)
        raise NotImplementedError(self.dist)


class ResidualBlock(nn.Module):
    """Residual block."""
    num_channels: int
    prefix: str
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        # Conv branch
        y = nn.relu(x)
        y = nn.Conv(self.num_channels,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding='SAME',
                    dtype=self.dtype,
                    kernel_init=default_conv_init(),
                    name=self.prefix + '/conv2d_1')(y)
        y = nn.relu(y)
        y = nn.Conv(self.num_channels,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding='SAME',
                    dtype=self.dtype,
                    kernel_init=default_conv_init(),
                    name=self.prefix + '/conv2d_2')(y)

        return y + x


class ResidualBlockDeconv(nn.Module):
    """Residual block."""
    num_channels: int
    prefix: str
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        # Conv branch
        y = nn.relu(x)
        y = nn.ConvTranspose(self.num_channels,
                             kernel_size=[3, 3],
                             strides=(1, 1),
                             padding='SAME',
                             dtype=self.dtype,
                             kernel_init=default_conv_init(),
                             name=self.prefix + '/deconv2d_1')(y)
        y = nn.relu(y)
        y = nn.ConvTranspose(self.num_channels,
                             kernel_size=[3, 3],
                             strides=(1, 1),
                             padding='SAME',
                             dtype=self.dtype,
                             kernel_init=default_conv_init(),
                             name=self.prefix + '/deconv2d_2')(y)
        return y + x


class ImpalaDeconv(nn.Module):
    """IMPALA decoder architecture."""
    prefix: str
    last_channel: int
    dtype: Any = jnp.float32
    num_channels: Tuple[int, ...] = (64, 64, 64)
    num_blocks: Tuple[int, ...] = (4, 4, 4)

    @nn.compact
    def __call__(self, x):
        out = nn.ConvTranspose(self.num_channels[0],
                               kernel_size=[1, 1],
                               strides=[1, 1],
                               padding="SAME",
                               dtype=self.dtype,
                               kernel_init=default_conv_init(),
                               name=self.prefix + '/deconv2d_init')(x)
        for i, (num_channels, num_blocks) in enumerate(zip(
            self.num_channels,
            self.num_blocks
        )):
            for j in range(num_blocks):
                block = ResidualBlockDeconv(num_channels,
                                            prefix='residual_deconv_{}_{}'.format(i, j),
                                            dtype=self.dtype)
                out = block(out)
            if i == len(self.num_channels) - 1:
                # Map back to RGB
                num_channels = self.last_channel
                dtype = jnp.float32
            else:
                dtype = self.dtype
            conv = nn.ConvTranspose(num_channels,
                                    kernel_size=[3, 3],
                                    strides=(2, 2),
                                    padding='SAME',
                                    dtype=dtype,
                                    kernel_init=default_conv_init(),
                                    name=self.prefix + '/deconv2d_%d' % i)
            out = conv(out)

        return out


class Impala(nn.Module):
    """IMPALA architecture."""
    prefix: str
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        out = x
        for i, (num_channels, num_blocks) in enumerate([(64, 4), (64, 4),
                                                        (64, 4)]):
            conv = nn.Conv(num_channels,
                           kernel_size=[3, 3],
                           strides=(1, 1),
                           padding='SAME',
                           dtype=self.dtype,
                           kernel_init=default_conv_init(),
                           name=self.prefix + '/conv2d_%d' % i)
            out = conv(out)

            out = nn.max_pool(out,
                              window_shape=(3, 3),
                              strides=(2, 2),
                              padding='SAME')
            for j in range(num_blocks):
                block = ResidualBlock(num_channels,
                                      prefix='residual_{}_{}'.format(i, j),
                                      dtype=self.dtype, )
                out = block(out)
        out = out.reshape(out.shape[0], -1)
        out = nn.relu(out)
        out = nn.Dense(256,
                       kernel_init=default_mlp_init(),
                       dtype=self.dtype,
                       name=self.prefix + '/representation')(out)
        out = nn.relu(out)
        return out


def get_activation(name):
    if name == 'none':
        return lambda x: x
    if name == 'mish':
        return lambda x: x * jnp.tanh(nn.softplus(x))
    elif hasattr(nn, name):
        return getattr(nn, name)
    else:
        raise NotImplementedError(name)


class DanijarEncoder(nn.Module):
    """Original Dreamerv2 cnn architecture."""
    prefix: str
    norm: bool
    act: str
    cnn_kernels: Sequence[int] = (4, 4, 4, 4)
    cnn_depth: int = 4  # 48
    dtype: Any = jnp.float32

    # 64 32 16 8
    # 48 96 192 384

    @nn.compact
    def __call__(self, x):
        for i, kernel in enumerate(self.cnn_kernels):
            depth = 2 ** i * self.cnn_depth
            x = nn.Conv(depth,
                        kernel_size=[kernel, kernel],
                        strides=(2, 2),
                        padding='VALID',
                        kernel_dilation=(1, 1),
                        dtype=self.dtype,
                        kernel_init=default_conv_init(),
                        name=self.prefix + '/conv2d_%d' % i)(x)
            if self.norm:
                x = nn.LayerNorm(dtype=self.dtype)(x)
            x = get_activation(self.act)(x)
        return x


class DanijarDecoder(nn.Module):
    """Original Dreamerv2 cnn architecture."""
    prefix: str
    norm: bool
    act: str
    cnn_kernels: Sequence[int] = (5, 5, 6, 6)
    cnn_depth: int = 48
    last_channel: int = 3
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        for i, kernel in enumerate(self.cnn_kernels):
            if i == len(self.cnn_kernels) - 1:
                depth = self.last_channel
            else:
                depth = 2 ** (len(self.cnn_kernels) - i - 2) * self.cnn_depth
            x = nn.ConvTranspose(depth,
                                 kernel_size=[kernel, kernel],
                                 strides=(2, 2),
                                 padding='VALID',
                                 dtype=self.dtype,
                                 kernel_dilation=(1, 1),
                                 kernel_init=default_conv_init(),
                                 name=self.prefix + '/deconv2d_%d' % i)(x)
            if i != len(self.cnn_kernels) - 1:
                if self.norm:
                    x = nn.LayerNorm(dtype=self.dtype)(x)
                x = get_activation(self.act)(x)
        return x
