from typing import Any, Tuple

import jax.numpy as jnp
import flax.linen as nn
from jax.random import PRNGKey
from flax.core import FrozenDict
from functools import partial
import jax

import expl

from common.nets import EnsembleRSSM, Encoder, MLP, Decoder
from common.other import RollingNorm, action_noise, schedule
import optax
from flax.optim import DynamicScale
from flax.training import train_state
import flax
import functools
import time

import dm_pix as pix


class TrainState(train_state.TrainState):
    # batch_stats: Any
    dynamic_scale: flax.optim.DynamicScale


def h(x):
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + 0.001 * x


def h_inv(x):
    inner = ((jnp.sqrt(1 + 4 * 0.001 * (jnp.abs(x) + 1 + 0.001)) - 1) /
             (2 * 0.001)) ** 2 - 1
    return jnp.sign(x)*inner


def compute_grad_norm(grads):
    if hasattr(grads, "items"):
        acc = 0.
        n = 0
        for k, v in grads.items():
            acc += compute_grad_norm(v)
            n += 1
        acc /= n
    else:
        acc = jnp.linalg.norm(grads)
    return acc


def target_update(online, target, tau: float):
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), online, target)

    return new_target_params


@partial(jax.jit)
def state_update(state_online,
                 state_target,
                 tau: float = 1.):
    """ Update key weights as tau * online + (1-tau) * target
    """
    new_weights = target_update(state_online.params,
                                state_target.params, tau)
    new_params = state_target.params.copy(
        add_or_replace=new_weights)

    state_target = state_target.replace(params=new_params)
    return state_target


def critic_loss_fn(params, train_state_critic, value_fn, feat, weight, target, rescale_rewards_wm_critic):
    # States:     [z0]  [z1]  [z2]   z3
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]   v3
    # Weights:    [ 1]  [w1]  [w2]   w3
    # Targets:    [t0]  [t1]  [t2]
    # Loss:        l0    l1    l2
    if rescale_rewards_wm_critic == "muzero":
        target = h(target)
    dist = train_state_critic.apply_fn({'params': {'critic': params}}, feat[:-1], method=value_fn)
    # dist = self.critic(seq['feat'][:-1])
    target = jax.lax.stop_gradient(target)
    weight = jax.lax.stop_gradient(weight)
    critic_loss = -dist.log_prob(jnp.expand_dims(target, -1)) + jax.lax.stop_gradient(dist.log_prob(dist.mean()).mean())
    return (critic_loss * weight[:-1]).mean(), target.mean()


def nstep_target(params_target, apply_fn, apply_target_fn, lambda_return_fn, feat, reward, disc, discount_lambda, rescale_rewards_wm_critic):
    """ ALWAYS OUTPUTS TARGETS IN REGULAR SPACE
    APPLY h AGAIN IF NEEDED"""
    # States:     [z0]  [z1]  [z2]  [z3]
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]  [v3]
    # Discount:   [d0]  [d1]  [d2]   d3
    # Targets:     t0    t1    t2
    value = apply_fn({'params': {'target_critic': params_target}}, feat, method=apply_target_fn).mode()[:, :, 0]
    if rescale_rewards_wm_critic == 'muzero':
        # MuZero-style target rescaling
        value = h_inv(value)
        reward = h_inv(reward)
    # Skipping last time step because it is used for bootstrapping.
    bootstrap = value[-1]
    next_values = jnp.concatenate([value[1:-1], bootstrap[None]], 0)
    inputs = reward.transpose()[:-1] + disc[:-1] * next_values * (1 - discount_lambda)
    targets, acc = jax.lax.scan(lambda_return_fn, bootstrap,
                                (inputs, disc[:-1], jnp.ones_like(inputs) * discount_lambda), reverse=True)
    del targets
    return acc


@functools.partial(jax.jit, static_argnames=(
        'wm',
        'ac',
        'config',
        'preprocess',
        'encode',
        'decode',
        'observe',
        'get_feat',
        'img_step',
        'decode_reward',
        'decode_discount',
        'kl_loss',
        "policy_fn",
        "value_fn",
        "normalize_rew_fn",
        "target_fn",
        "lambda_return_fn"
))
def train(obs,
          action,
          reward,
          is_first,
          is_terminal,
          train_state_wm,
          train_state_actor,
          train_state_critic,
          train_state_target,
          preprocess,
          encode,
          decode,
          observe,
          get_feat,
          img_step,
          decode_reward,
          decode_discount,
          kl_loss,
          policy_fn,
          value_fn,
          normalize_rew_fn,
          target_fn,
          lambda_return_fn,
          config,
          step,
          stats,
          stoch,
          det,
          rng):
    # 1. Train WM
    start_time = time.time()
    train_state_wm, (model_loss, (stoch_post, deter,
                                  stats_post, loss_r, loss_d,
                                  loss_rec, kl_loss, rec_obs), grads_wm) = \
        opt_wrapper(wm_loss_fn,
                    train_state_wm,
                    config["half_precision"],
                    (
                        train_state_wm.params,
                        train_state_wm,
                        preprocess,
                        encode,
                        decode,
                        observe,
                        get_feat,
                        decode_reward,
                        decode_discount,
                        kl_loss,
                        config["grad_heads"],
                        config["pred_discount"],
                        config["loss_scales"]['kl'],
                        config["loss_scales"]['reward'],
                        config["loss_scales"]['discount'],
                        config["loss_scales"]['proprio'],
                        config['rescale_rewards_wm_critic'],
                        rng,
                        obs,
                        action,
                        reward,
                        is_first,
                        is_terminal,
                        stats,
                        stoch,
                        det,))

    print("WM train took {}".format(time.time() - start_time))

    # 2. Train AC
    start = (stats_post, stoch_post, deter)
    start = tuple(t.reshape(-1, *t.shape[2:]) for t in start)
    start_time = time.time()
    train_state_actor, (
        actor_loss, (feat_acc, action_acc, disc_acc, ent, weight, state, img_reward, target), grads_actor) = \
        opt_wrapper(actor_loss_fn, train_state_actor, config["half_precision"],
                    (
                        train_state_actor.params,
                        train_state_actor,
                        train_state_critic,
                        train_state_wm,
                        img_step,
                        decode_reward,
                        decode_discount,
                        get_feat,
                        policy_fn,
                        value_fn,
                        normalize_rew_fn,
                        target_fn,
                        lambda_return_fn,
                        config["discount"],
                        config["pred_discount"],
                        config["imag_horizon"],
                        config["rssm"]["ensemble"],
                        config["actor_grad"],
                        config["actor_ent"],
                        step,
                        config["discount_lambda"],
                        rng,
                        start,
                        is_terminal,
                        config['rescale_rewards_actor'],
                        config['rescale_rewards_wm_critic'],
                        config['actor_grad_mix'])
                    )

    # self.train_state_actor = self.train_state_actor.apply_gradients(grads=grads_actor)
    print("Actor train took {}".format(time.time() - start_time))

    start_time = time.time()
    train_state_critic, (critic_loss, critic_targets, grads_critic) = opt_wrapper(
        critic_loss_fn,
        train_state_critic,
        config["half_precision"],
        (train_state_critic.params,
         train_state_critic,
         value_fn,
         feat_acc,
         weight,
         target,
         config['rescale_rewards_wm_critic']))

    print("Critic train took {}".format(time.time() - start_time))

    train_state_target = state_update(train_state_critic,
                                      train_state_target,
                                      config["slow_target_fraction"])

    wm_gradnorms = {"gradnorm_{}".format(k): compute_grad_norm(grad) for k, grad in grads_wm.items()}
    metrics = {
        'loss_reward': loss_r,
        'loss_discount': loss_d,
        'loss_reconstruction': loss_rec,
        'loss_kl': kl_loss,
        'loss_actor': actor_loss,
        'loss_critic': critic_loss,
        **wm_gradnorms,
        'gradnorm_actor': compute_grad_norm(grads_actor),
        'gradnorm_critic': compute_grad_norm(grads_critic),
        'policy_entropy': ent.mean(),
        'mean_img_action_var': action_acc.reshape(-1, action.shape[-1]).var(0).mean(),
        'mean_img_reward': img_reward.sum(-1).mean(),
        'mean_img_nstep_targets': target.mean(),
        'mean_true_reward': reward.sum(-1).mean()
    }
    return (train_state_wm, train_state_actor, train_state_critic, train_state_target), \
           (stats_post[:, -1], stoch_post[:, -1], deter[:, -1]), metrics, rec_obs


def opt_wrapper(loss_fn, train_state, half_precision, args):
    if half_precision:
        grad_fn = train_state.dynamic_scale.value_and_grad(
            loss_fn, has_aux=True)
        dynamic_scale, is_fin, (loss, aux), grads = grad_fn(*args)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(*args)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        # grads = jax.lax.pmean(grads, axis_name='batch')
        grads = jax.tree_util.tree_map(jnp.nan_to_num, grads)
        dynamic_scale = None

    new_state = apply_grads(grads, train_state, dynamic_scale=dynamic_scale)

    if half_precision:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        new_state = new_state.replace(
            opt_state=jax.tree_multimap(
                functools.partial(jnp.where, is_fin),
                new_state.opt_state,
                train_state.opt_state),
            params=jax.tree_multimap(
                functools.partial(jnp.where, is_fin),
                new_state.params,
                train_state.params))

    return new_state, (loss, aux, grads)


@jax.jit
def apply_grads(grads, state, dynamic_scale=None):
    return state.apply_gradients(grads=grads, dynamic_scale=dynamic_scale)


def wm_loss_fn(
        params,
        train_state_wm,
        preprocess_fn,
        encode_fn,
        decode_fn,
        observe_fn,
        feature_fn,
        decode_reward_fn,
        decode_discount_fn,
        kl_loss_fn,
        grad_heads,
        pred_discount,
        kl_loss_scale,
        reward_loss_scale,
        discount_loss_scale,
        reconstr_loss_scale,
        rescale_rewards_wm_critic,
        rng,
        obs,
        action,
        reward,
        is_first,
        is_terminal,
        stats, stoch, det):
    rng, key = jax.random.split(rng)
    
    obs, action, reward, is_first, is_terminal, discount = train_state_wm.apply_fn({'params': params},
                                                                                   obs, action, reward,
                                                                                   is_first,
                                                                                   is_terminal, key, 
                                                                                   rescale_rewards_wm_critic,
                                                                                   method=preprocess_fn)
    embed = train_state_wm.apply_fn({'params': params},
                                    obs, reward, method=encode_fn)
    rng, key = jax.random.split(rng)
    (stats_post, stoch_post, deter), (stats_prior, stoch_prior, deter) = train_state_wm.apply_fn({'params': params},
                                                                                                 embed, action,
                                                                                                 is_first, key, stats,
                                                                                                 stoch, det,
                                                                                                 method=observe_fn)
    kl_loss, kl_value = train_state_wm.apply_fn({'params': params},
                                                (stats_post, stoch_post, deter), (stats_prior, stoch_prior, deter),
                                                method=kl_loss_fn)
    feat = train_state_wm.apply_fn({'params': params},
                                   stoch_post, deter, method=feature_fn)

    # reconstruction head
    inp = feat if 'decoder' in grad_heads else jax.lax.stop_gradient(feat)
    dist_rec = train_state_wm.apply_fn({'params': params},
                                   inp, method=decode_fn)

    like_rec = dist_rec.log_prob(obs.reshape(*dist_rec.batch_shape, *dist_rec.event_shape)) # [:,:,:,:,-3:]
    loss_rec = -like_rec.mean() + jax.lax.stop_gradient(dist_rec.log_prob(dist_rec.mean()).mean())

    # reward head
    inp = feat if 'reward' in grad_heads else jax.lax.stop_gradient(feat)
    dist = train_state_wm.apply_fn({'params': params},
                                   inp, method=decode_reward_fn)
    like_r = dist.log_prob(jnp.expand_dims(reward, -1))
    loss_r = -like_r.mean() + jax.lax.stop_gradient(dist.log_prob(dist.mean()).mean())

    if pred_discount:
        inp = feat if 'discount' in grad_heads else jax.lax.stop_gradient(feat)
        dist = train_state_wm.apply_fn({'params': params},
                                       inp, method=decode_discount_fn)
        like_d = dist.log_prob(jnp.expand_dims(discount, -1))
        loss_d = -like_d.mean() + jax.lax.stop_gradient(dist.log_prob(dist.mean()).mean())
    else:
        loss_d = 0.

    model_loss = reward_loss_scale * loss_r + discount_loss_scale * loss_d + reconstr_loss_scale * loss_rec + kl_loss_scale * kl_loss

    return model_loss, (stoch_post, deter, stats_post, loss_r, loss_d, loss_rec, kl_loss, dist_rec.mean())


def pred_rewards(apply_fn_wm,
                 decode_reward_fn,
                 feat_acc,
                 params_wm, ):
    reward = apply_fn_wm({'params': params_wm}, feat_acc, method=decode_reward_fn)
    reward = reward.mean()[:, :, 0].transpose()
    return reward


def target_pred(params_target, params_actor,
                feat_acc, reward,
                disc_acc,
                apply_fn,
                apply_fn_target,
                target_fn,
                policy_fn,
                lambda_return_fn,
                discount_lambda,
                rescale_rewards_wm_critic,
                key,
                ):
    returns = nstep_target(params_target, apply_fn_target, target_fn, lambda_return_fn, feat_acc, reward, disc_acc,
                           discount_lambda, rescale_rewards_wm_critic)
    policy = apply_fn({'params': {'actor': params_actor}}, jax.lax.stop_gradient(feat_acc[:-2]), method=policy_fn)
    return returns, policy, policy.entropy(key)


def actor_loss_fn(
        params,
        train_state_actor,
        train_state_target,
        train_state_wm,
        imagine_fn,
        decode_reward_fn,
        decode_discount_fn,
        feature_fn,
        policy_fn,
        value_fn,
        normalize_rew,
        target_fn,
        lambda_return_fn,
        discount,
        pred_discount,
        hor,
        ensemble,
        actor_grad,
        actor_ent,
        step,
        discount_lambda,
        rng,
        start,
        is_terminal,
        rescale_rewards_actor,
        rescale_rewards_wm_critic,
        actor_grad_mix):
    """
        Return should be of form:
        (state, feat, action, [optional: discount], reward)
    """
    feat_acc, action_acc, disc_acc, weight, state = imagine(params,
                                                            train_state_wm.params,
                                                            train_state_actor.apply_fn,
                                                            train_state_wm.apply_fn,
                                                            imagine_fn,
                                                            decode_discount_fn,
                                                            feature_fn,
                                                            policy_fn,
                                                            value_fn,
                                                            discount,
                                                            pred_discount,
                                                            hor,
                                                            ensemble,
                                                            rng,
                                                            start, is_terminal)

    reward = pred_rewards(train_state_wm.apply_fn, decode_reward_fn, feat_acc, train_state_wm.params)

    targets, policy, ent = target_pred(train_state_target.params,
                                       params,
                                       feat_acc, reward,
                                       disc_acc,
                                       train_state_actor.apply_fn,
                                       train_state_target.apply_fn,
                                       target_fn,
                                       policy_fn,
                                       lambda_return_fn,
                                       discount_lambda,
                                       rescale_rewards_wm_critic,
                                       rng,
                                       )
    if rescale_rewards_actor == 'muzero':
        # MuZero-style target rescaling
        actor_targets = h(targets)
    else:
        actor_targets = targets

    start = time.time()
    if actor_grad == 'dynamics':
        objective = actor_targets[1:]
    elif actor_grad == 'reinforce':
        baseline = train_state_target.apply_fn({'params': {'target_critic': train_state_target.params}}, feat_acc[:-2],
                                               method=target_fn).mode()[:, :, 0]
        advantage = jax.lax.stop_gradient(actor_targets[1:] - baseline)
        objective = policy.log_prob(action_acc[1:-1]) * advantage[:-1]
    elif actor_grad == 'mixture':
        baseline = train_state_target.apply_fn({'params': {'target_critic': train_state_target.params}}, feat_acc[:-2],
                                               method=target_fn).mode()[:, :, 0]
        advantage = jax.lax.stop_gradient(actor_targets[1:] - baseline)
        
        objective = policy.log_prob(action_acc[1:-1]) * advantage
        mix = actor_grad_mix
        objective = mix * actor_targets[1:] + (1 - mix) * objective
    else:
        raise NotImplementedError(actor_grad)
    print("Objective selection took {}".format(time.time() - start))

    # ent_scale = 1.
    # TODO: for now coeff to 1
    ent_scale = actor_ent  # schedule(actor_ent, step)
    objective += ent_scale * ent
    weight = jax.lax.stop_gradient(weight)

    actor_loss = -(weight[:-2] * objective).mean()

    return actor_loss, (feat_acc, action_acc, disc_acc, ent, weight, state, reward, targets)


def imagine(params_actor,
            params_wm,
            apply_fn,
            apply_fn_wm,
            imagine_fn,
            decode_discount_fn,
            feature_fn,
            policy_fn,
            value_fn,
            discount,
            pred_discount,
            horizon,
            ensemble,
            rng,
            state,
            is_terminal):
    """
    Return should be of form:
    (state, feat, action, [optional: discount])
    """
    
    feat = apply_fn_wm({'params': params_wm}, state[1], state[2], method=feature_fn)
    rng, key = jax.random.split(rng)
    action = apply_fn({'params': {'actor': params_actor}}, feat, method=policy_fn).mode(key)
    feat_acc, action_acc = [feat], [action]
    for _ in range(horizon):
        rng, key = jax.random.split(rng)
        action = apply_fn({'params': {'actor': params_actor}}, jax.lax.stop_gradient(feat_acc[-1]),
                          method=policy_fn).sample(seed=key)
        rng, key = jax.random.split(rng)
        state = apply_fn_wm({'params': params_wm}, state[0], state[1], state[2], action, key, sample=True,
                            method=imagine_fn)
        feat = apply_fn_wm({'params': params_wm}, state[1], state[2], method=feature_fn)
        feat_acc.append(feat)
        action_acc.append(action)
    feat_acc = jnp.stack(feat_acc, 0)
    action_acc = jnp.stack(action_acc, 0)
    if pred_discount:
        disc = apply_fn_wm({'params': params_wm}, feat_acc, method=decode_discount_fn).mean()
        if is_terminal is not None:
            # Override discount prediction for the first step with the true
            # discount factor from the replay buffer.
            true_first = (1.0 - (is_terminal).reshape(1, -1)) * discount
            disc = jnp.concatenate([true_first, disc[1:, :, 0]], 0)
    else:
        disc = discount * jnp.ones(feat_acc.shape[:-1])

    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.
    weight = jnp.cumprod(
        jnp.concatenate([jnp.ones_like(disc[:1]), disc[:-1]], 0), 0)

    return feat_acc, action_acc, disc, weight, state


@functools.partial(jax.jit, static_argnames=[
    "reset",
    "sample_latent",
    "sample_action",
    "wm_init_fn",
    "wm_encode_fn",
    "wm_obs_step_fn",
    "wm_get_feat_fn",
    "ac_policy_fn",
    "apply_fn_wm",
    "apply_fn_actor",
    "act_shape",
    "discrete_actions",
    "noise",
])
def apply_policy(obs, is_first, state, reward,
                 act_shape,
                 discrete_actions,
                 reset,
                 apply_fn_wm,
                 apply_fn_actor,
                 wm_init_fn,
                 wm_encode_fn,
                 wm_obs_step_fn,
                 wm_get_feat_fn,
                 ac_policy_fn,
                 wm_params,
                 ac_params,
                 rng,
                 noise,
                 sample_latent,
                 sample_action,
                 ):
    key, rng = jax.random.split(rng)
    if reset:
        stats, stoch, det = apply_fn_wm({'params': wm_params}, key,
                                        method=wm_init_fn)
        # Hack to init latent with only 1 sample
        stats = stats[:1]
        stoch = stoch[:1]
        det = det[:1]
        action = jnp.zeros((len(obs),) + act_shape)
        state = (stats, stoch, det), action
    latent, action = state
    if obs.dtype == jnp.uint8:
        obs = obs.astype(jnp.float32) / 255.0 - 0.5
    embed = apply_fn_wm({'params': wm_params}, obs, reward, method=wm_encode_fn)

    key, rng = jax.random.split(rng)

    latent, _ = apply_fn_wm({'params': wm_params}, *latent, action, embed[:, 0],
                            is_first, key, sample_latent, method=wm_obs_step_fn)

    feat = apply_fn_wm({'params': wm_params}, latent[1], latent[2],
                       method=wm_get_feat_fn)

    key, rng = jax.random.split(rng)
    actor = apply_fn_actor({'params': {'actor': ac_params}}, feat,
                           method=ac_policy_fn)
    if sample_action:
        action = actor.mode(key)
    else:
        action = actor.sample(seed=key)
    action = action_noise(action, noise, discrete_actions)
    return latent, action, rng


class Agent:
    def __init__(self, config, obs_space, act_space, step):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.step = step
        self.updates = 0

        self.half_precision = config.half_precision
        if config.half_precision:
            self.dtype = jnp.float16
        else:
            self.dtype = jnp.float32

        self.wm = WorldModel(config.dataset.batch,
                             config.pred_discount,
                             config.discount,
                             config.clip_rewards,
                             config.seed,
                             config.data_aug,
                             config.framestack,
                             FrozenDict({k: tuple(v.shape) for k, v in obs_space.items()}),
                             FrozenDict(config.rssm),
                             FrozenDict(config.encoder),
                             FrozenDict(config.decoder),
                             FrozenDict(config.reward_head),
                             FrozenDict(config.discount_head),
                             FrozenDict(config.kl),
                             dtype=self.dtype,
                             )

        discrete_actions = hasattr(self.act_space, 'n')
        if self.config.actor_grad == 'auto':
            self.config = self.config.update({
                'actor_grad':
                    'reinforce' if discrete_actions else 'dynamics'
            })
        if self.config.actor['dist'] == 'auto':
            self.config = self.config.update({
                'actor.dist':
                    'onehot' if discrete_actions else 'trunc_normal'
            })

        self.ac = ActorCritic(config.dataset.batch, config.rssm.stoch,
                              config.rssm.discrete, config.rssm.gru_hidden,
                              config.slow_target,
                              config.imag_horizon, config.seed,
                              self.act_space.shape[0],
                              FrozenDict(config.reward_norm),
                              FrozenDict(self.config.actor),
                              FrozenDict(self.config.critic),
                              dtype=self.dtype)
        self.rng = jnp.array(PRNGKey(config.seed + 1))
        self.setup()
        if config.expl_behavior == 'greedy':
            self._expl_behavior = self.ac
        else:
            self._expl_behavior = getattr(expl, config.expl_behavior)(
                self.config, self.act_space, self.wm, self.tfstep,
                lambda seq: self.wm.reward_pred(seq['feat']).mode())

    def setup(self):
        init_batch_size = 1
        key, self.rng = jax.random.split(self.rng)
        if self.config.obs_type == 'pixels':
            dummy_obs_seq = jax.random.uniform(key,
                                               [init_batch_size, self.config.dataset.length] +
                                               list(self.obs_space[
                                                        'image'].shape))  # [self.config.framestack*self.obs_space['image'].shape[-1]]
        elif self.config.obs_type == 'full_state':
            dummy_obs_seq = jax.random.uniform(key,
                                               [init_batch_size, self.config.dataset.length,
                                                self.obs_space['image'].shape[-1]])
        key, self.rng = jax.random.split(self.rng)
        dummy_rew_seq = jax.random.uniform(key,
                                           [init_batch_size, self.config.dataset.length, 1])
        key, self.rng = jax.random.split(self.rng)
        dummy_act_seq = jax.random.uniform(key,
                                           [init_batch_size, self.config.dataset.length] +
                                           list(self.act_space.shape))
        dummy_is_first_seq = jnp.zeros(
            [init_batch_size, self.config.dataset.length])
        key, self.rng = jax.random.split(self.rng)
        self.wm_params = self.wm.init(key, dummy_obs_seq, dummy_rew_seq, dummy_act_seq,
                                      dummy_is_first_seq, key, batch_size=init_batch_size)['params']
        self.wm_opt = optax.chain(
            optax.clip_by_global_norm(self.config.model_opt.clip),
            optax.adamw(self.config.model_opt.lr,
                        eps=self.config.model_opt.eps,
                        weight_decay=self.config.model_opt.wd))

        if self.half_precision:
            dynamic_scale_wm = DynamicScale()

            # NAN can occur in the actor for other reasons,
            # so don't punish it too much
            dynamic_scale_actor = DynamicScale(growth_interval=100)
            dynamic_scale_critic = DynamicScale()
        else:
            dynamic_scale_wm = dynamic_scale_actor = dynamic_scale_critic = None

        self.train_state_wm = TrainState.create(apply_fn=self.wm.apply,
                                                params=self.wm_params,
                                                tx=self.wm_opt,
                                                dynamic_scale=dynamic_scale_wm)

        self.actor_opt = optax.chain(
            optax.clip_by_global_norm(self.config.actor_opt.clip),
            optax.adamw(self.config.actor_opt.lr,
                        eps=self.config.actor_opt.eps,
                        weight_decay=self.config.actor_opt.wd))
        self.critic_opt = optax.chain(
            optax.clip_by_global_norm(self.config.critic_opt.clip),
            optax.adamw(self.config.critic_opt.lr,
                        eps=self.config.critic_opt.eps,
                        weight_decay=self.config.critic_opt.wd))
        self.target_opt = optax.chain(
            optax.clip_by_global_norm(self.config.critic_opt.clip),
            optax.adamw(self.config.critic_opt.lr,
                        eps=self.config.critic_opt.eps,
                        weight_decay=self.config.critic_opt.wd))

        key, self.rng = jax.random.split(self.rng)
        dummy_seq = (jax.random.uniform(key,
                        (self.config.imag_horizon,
                         init_batch_size,
                         self.config.rssm.stoch * self.config.rssm.discrete
                         + self.config.rssm.gru_hidden * \
                         self.config.rssm.gru_layers)))

        self.ac_params = self.ac.init({"params": key}, dummy_seq, True)['params']

        self.train_state_actor = TrainState.create(apply_fn=self.ac.apply,
                                                   params=self.ac_params['actor'],
                                                   tx=self.actor_opt,
                                                   dynamic_scale=dynamic_scale_actor)
        self.train_state_critic = TrainState.create(apply_fn=self.ac.apply,
                                                    params=self.ac_params['critic'],
                                                    tx=self.critic_opt,
                                                    dynamic_scale=dynamic_scale_critic)
        self.train_state_target = TrainState.create(apply_fn=self.ac.apply,
                                                    params=self.ac_params['target_critic'],
                                                    dynamic_scale=dynamic_scale_critic,
                                                    tx=self.target_opt)

    def init_policy_state(self, n):
        key, rng = jax.random.split(self.rng)
        stats, stoch, det = self.train_state_wm.apply_fn({'params': self.train_state_wm.params}, key,
                                                         method=self.wm.init_rssm)
        # Hack to init latent with only 1 sample
        stats = stats[:n]
        stoch = stoch[:n]
        det = det[:n]
        action = jnp.zeros((n,) + self.act_space.shape)
        state = (stats, stoch, det), action
        return state

    def policy(self, obs, is_first, state, reward, reset=False, mode='train'):
        if mode == 'eval':
            noise = self.config.eval_noise
        elif mode == 'explore':
            noise = self.config.expl_noise
        elif mode == 'train':
            noise = self.config.expl_noise

        if self.config['rescale_rewards_wm_critic'] == 'muzero':
            # MuZero-style target rescaling
            reward = h(reward)

        latent, action, self.rng = apply_policy(obs, is_first, state, reward,
                                                self.act_space.shape,
                                                hasattr(self.act_space, "n"),
                                                reset,
                                                self.train_state_wm.apply_fn,
                                                self.train_state_actor.apply_fn,
                                                self.wm.init_rssm,
                                                self.wm.encode,
                                                self.wm.obs_step,
                                                self.wm.get_feat,
                                                self.ac.policy,
                                                self.train_state_wm.params,
                                                self.train_state_actor.params,
                                                self.rng,
                                                noise,
                                                sample_latent=(mode == 'train') or not self.config.eval_state_mean,
                                                sample_action=(mode != 'eval'),
                                                )
        state = (latent, action)
        return action, state

    def train(self,
              obs,
              action,
              reward,
              is_first,
              is_terminal,
              stats,
              stoch,
              det,
              reinit=False):
        # 1. Train WM

        if reinit:
            stats, stoch, det = self.wm.init_rssm(self.rng)
        self.rng, key = jax.random.split(self.rng)

        (self.train_state_wm, self.train_state_actor, self.train_state_critic,
         self.train_state_target), carry, metrics, rec_obs = train(
            obs,
            action,
            reward,
            is_first,
            is_terminal,
            train_state_wm=self.train_state_wm,
            train_state_actor=self.train_state_actor,
            train_state_critic=self.train_state_critic,
            train_state_target=self.train_state_target,
            preprocess=self.wm.preprocess,
            encode=self.wm.encode,
            decode=self.wm.decode,
            observe=self.wm.observe,
            get_feat=self.wm.get_feat,
            img_step=self.wm.img_step,
            decode_reward=self.wm.decode_reward,
            decode_discount=self.wm.decode_discount,
            kl_loss=self.wm.kl_loss,
            policy_fn=self.ac.policy,
            value_fn=self.ac.value,
            normalize_rew_fn=self.ac.normalize_rew,
            target_fn=self.ac.target,
            lambda_return_fn=self.ac.lambda_return,
            config=FrozenDict(self.config),
            step=self.step.value,
            stats=stats,
            stoch=stoch,
            det=det,
            rng=key,
        )
        self.updates += 1
        metrics_ = {}
        for k,v in metrics.items():
            try:
                metrics_[k] = v.item()
            except:
                metrics_[k] = v
        metrics = metrics_

        if self.config.half_precision:
            metrics["loss_scale_wm"] = self.train_state_wm.dynamic_scale.scale.item()
            metrics["loss_scale_actor"] = self.train_state_actor.dynamic_scale.scale.item()
            metrics["loss_scale_critic"] = self.train_state_critic.dynamic_scale.scale.item()

        return carry, metrics, rec_obs

    def report(self, data):
        report = {}
        data = self.wm.preprocess(data)
        for key in self.wm.heads['decoder'].cnn_keys:
            name = key.replace('/', '_')
            report[f'openl_{name}'] = self.wm.video_pred(data, key)
        return report


class WorldModel(nn.Module):
    batch: int
    pred_discount: bool
    discount: float
    clip_rewards: str
    seed: int
    data_aug: bool
    framestack: int
    obs_shapes: FrozenDict
    config_rssm: FrozenDict
    config_encoder: FrozenDict
    config_decoder: FrozenDict
    config_reward_head: FrozenDict
    config_discount_head: FrozenDict
    config_kl: FrozenDict
    dtype: Any = jnp.float32
    obs_type: str = "pixels"

    def setup(self):
        self.rssm = EnsembleRSSM(**self.config_rssm,
                                 batch=self.batch,
                                 seed=self.seed,
                                 dtype=self.dtype,
                                 config_kl=self.config_kl)
        self.encoder = Encoder(self.obs_shapes, dtype=self.dtype, **self.config_encoder, obs_type=self.obs_type)
        self.decoder = Decoder(self.obs_shapes, dtype=self.dtype, **self.config_decoder, channels=3,
                               framestack=self.framestack, obs_type=self.obs_type)
        self.reward_pred = MLP(out_shape=(1,), dtype=self.dtype, **self.config_reward_head)
        if self.pred_discount:
            self.discount_pred = MLP(out_shape=(1,),
                                     dtype=self.dtype,
                                     **self.config_discount_head)

    @nn.compact
    def __call__(self, obs, reward, action, is_first, rng, batch_size=None):
        key, rng = jax.random.split(rng)
        obs_z = self.encoder(obs, reward)

        ((stats_post, stoch_post, deter),
         (stats_prior, stoch_prior, det_prior)), \
        (kl_loss, kl_value) = self.rssm(obs_z,
                                        action,
                                        is_first,
                                        key,
                                        False,
                                        batch_size)
        feat = self.get_feat(stoch_post, deter)

        x_hat = self.decoder(feat)

        r_hat = self.reward_pred(feat)

        if self.pred_discount:
            discount_hat = self.discount_pred(feat)
        else:
            discount_hat = None
        return obs_z, feat, x_hat, r_hat, discount_hat

    def encode(self, obs, reward=None, action=None):
        latent = self.encoder(obs, reward, action)
        return latent

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def decode_reward(self, z):
        r_hat = self.reward_pred(z)
        return r_hat

    def decode_discount(self, z):
        assert self.pred_discount
        discount_hat = self.discount_pred(z)
        return discount_hat

    def observe(self, embed, action, is_first, rng, stats, stoch, det):
        return self.rssm.observe(embed, action, is_first, rng, stats, stoch, det)

    def kl_loss(self, post, prior):
        return self.rssm.kl_loss(post, prior)

    def get_feat(self, stoch, deter):
        return self.rssm.get_feat(stoch, deter)

    def img_step(self, prev_stats, prev_stoch, prev_det, prev_action, rng, sample):
        return self.rssm.img_step(prev_stats, prev_stoch, prev_det, prev_action, rng, sample=sample)

    def obs_step(self, prev_stats, prev_stoch, prev_det, prev_action, embed, is_first, rng, sample=True):
        return self.rssm.obs_step(prev_stats, prev_stoch, prev_det, prev_action, embed, is_first, rng, sample=sample)

    def init_rssm(self, key):
        return self.rssm.init_rssm(key)

    def preprocess(self, obs, action, reward, is_first, is_terminal, rng, rescale_rewards_wm_critic):
        return preprocess(obs, action, reward, is_first,
                          is_terminal, self.clip_rewards, self.discount, rng, rescale_rewards_wm_critic, self.data_aug)


@functools.partial(jax.jit, static_argnames=("clip_rewards", "data_aug", "rescale_rewards_wm_critic"))
def preprocess(obs, action, reward, is_first,
               is_terminal, clip_rewards, discount, rng, rescale_rewards_wm_critic, data_aug=False):
    if data_aug:
        aug_state = jnp.pad(obs, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]], 'edge')
        aug_state = pix.random_crop(key=rng, image=aug_state, crop_sizes=obs.shape)
    if obs.dtype == jnp.uint8:
        obs = obs.astype(jnp.float32) / 255.0 - 0.5
    if action.dtype == jnp.int32:
        action = action.astype(jnp.float32)

    if clip_rewards == 'sign':
        reward = nn.soft_sign(reward)
    elif clip_rewards == 'tanh':
        reward = nn.tanh(reward)
    if rescale_rewards_wm_critic == 'muzero':
        # MuZero-style target rescaling
        reward = h(reward)
    discount = (1.0 -
                is_terminal.astype(jnp.float32)) * discount
    return obs, action, reward, is_first, is_terminal, discount


class ActorCritic(nn.Module):
    batch: int
    stoch: int
    discrete: int
    gru_hidden: int
    slow_target: bool
    imag_horizon: int
    seed: int
    n_actions: int
    reward_norm: FrozenDict
    config_actor: FrozenDict
    config_critic: FrozenDict
    dtype: Any = jnp.float32

    def setup(self):
        self.actor = MLP((self.n_actions,), dtype=self.dtype, **self.config_actor)
        self.critic = MLP((1,), **self.config_critic)
        if self.slow_target:
            self.target_critic = MLP((1,), dtype=self.dtype, **self.config_critic)
            self.updates = 0
        else:
            self.target_critic = self.critic
        self.rewnorm = RollingNorm(**self.reward_norm)

    @nn.compact
    def __call__(self, seq, target=False):
        actions = self.actor(seq)
        values = self.critic(seq)
        if target:
            _ = self.target_critic(seq)
        return actions, values

    def normalize_rew(self, x):
        return self.rewnorm(x)

    def target(self, x):
        return self.target_critic(x)

    def policy(self, x):
        return self.actor(x)

    def value(self, x):
        return self.critic(x)

    def lambda_return(self, state, inputs):
        # Setting lambda=1 gives a discounted Monte Carlo return.
        # Setting lambda=0 gives a fixed 1-step return.
        inputs_, discount, lambda_ = inputs
        return inputs_ + discount * lambda_ * state, inputs_ + discount * lambda_ * state
