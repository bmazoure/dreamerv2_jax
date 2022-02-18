import argparse
import collections
import logging
import os
import pathlib
import re
import sys
import warnings

import numpy as np
import jax.numpy as jnp
import ruamel.yaml as yaml
import wandb
import time
import jax


sys.path.append("dreamerv2/")
import agent
import common

import cProfile

parser = argparse.ArgumentParser(description='Argparser')

try:
    import rich.traceback
    rich.traceback.install()
except ImportError:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))


def main(config):
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    # config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)

    metrics = collections.defaultdict(list)

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_video_train = common.Every(config.eval_every)
    should_video_eval = common.Every(config.eval_every)
    should_expl = common.Until(config.expl_until)

    if not config.jit:
        jax.config.update('jax_disable_jit', True)

    def make_env(mode):
        suite, task = config.task.split('_', 1)
        if suite == 'dmc':
            env = common.DMC(task, config.action_repeat, config.render_size,)
                            # config.dmc_camera)
            env = common.NormalizeAction(env, key=None)
        elif suite == 'atari':
            env = common.Atari(task, config.action_repeat, config.render_size,
                            config.atari_grayscale)
            env = common.OneHotAction(env)
        elif suite == 'crafter':
            assert config.action_repeat == 1
            outdir = logdir / 'crafter' if mode == 'train' else None
            reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
            env = common.Crafter(outdir, reward)
            env = common.OneHotAction(env)
        elif suite == "mw":
            env = common.MetaWorld(name=task,
                                   obs_type=config.obs_type,
                                   type_='train',
                                   dims=[config.render_size[0],
                                         config.render_size[1]],
                                   n_channels=3,
                                   camera=config.dmc_camera,
                                   start_task=config.start_task,
                                   n_tasks=config.n_tasks)
            env = common.NormalizeAction(env, key=None)
        else:
            raise NotImplementedError(suite)
        if config.framestack > 1:
            env = common.FrameStackWrapper(env, n_frames=config.framestack)
        if config.time_limit > 0:
            env = common.TimeLimit(env, config.time_limit)
        env = common.envs.DelayedResetWrapper(env)
        return env

    def per_episode(infos, mode):
        ep = {k: np.stack([info[k] for info in infos], 0) for k in infos[0].keys()}
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        to_log = {}
        print(
            f'{mode.title()} episode has {length} steps and return {score:.1f}.'
        )
        to_log[f'{mode}/return'] = score
        to_log[f'{mode}/length'] = length
        # logger.scalar(f'{mode}_return', score)
        # logger.scalar(f'{mode}_length', length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                # logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
                to_log[f'{mode}/sum/{key}'] = ep[key].sum()
            if re.match(config.log_keys_mean, key):
                # logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
                to_log[f'{mode}/mean/{key}'] = ep[key].mean()
            if re.match(config.log_keys_max, key):
                # logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
                to_log[f'{mode}/max/{key}'] = ep[key].max(0).mean()
        
        to_log[f'{mode}/success'] = to_log[f'{mode}/max/log_success']
        to_log[f'{mode}/reward'] = to_log[f'{mode}/mean/log_reward']
        to_log[f'{mode}/task_id'] = to_log[f'{mode}/mean/log_task']
        del to_log[f'{mode}/max/log_success'], to_log[f'{mode}/mean/log_reward'], to_log[f'{mode}/mean/log_task'], to_log[f'{mode}/mean/log_success']
        
        for key, value in to_log.items():
            if 'success' in key and 'log' in key:
                task_success_key = key
        to_log['%s/success_task_%s' % (mode,task_success_key.split('/')[-1].split('_')[1])] = to_log[task_success_key]
        del to_log[task_success_key]
        
        wandb.log(to_log)

    print('Create envs.')
    num_eval_envs = min(config.envs, config.eval_eps)
    
    make_env_train = lambda: make_env("train")
    make_env_eval = lambda: make_env("eval")
    
    if config.async_envs:
        train_envs = common.InfoResetAsyncVectorEnv([make_env_train]*config.envs, shared_memory=True)
        eval_envs = common.InfoResetAsyncVectorEnv([make_env_eval]*num_eval_envs, shared_memory=True)
    else:
        train_envs = common.InfoResetSyncVectorEnv([make_env_train]*config.envs)
        eval_envs = common.InfoResetSyncVectorEnv([make_env_eval]*num_eval_envs)
    train_envs.seed(config.seed)
    eval_envs.seed(config.seed+config.envs)

    act_space = train_envs.act_space
    obs_space = train_envs.obs_space

    if config.replay.prioritized:
        buffer_cls = common.PrioritizedJaxSubsequenceParallelEnvReplayBuffer
    else:
        buffer_cls = common.JaxSubsequenceParallelEnvReplayBuffer

    no_framestack_shape = list(train_envs.obs_space["image"].shape)
    no_framestack_shape[-1] = no_framestack_shape[-1] // config.framestack
    train_replay = buffer_cls(
        stack_size=config.framestack,
        n_envs=config.envs,
        batch_size=config.dataset.batch,
        subseq_len=config.dataset.length,
        persistent=config.replay.persistent,
        seed=config.seed,
        observation_shape=tuple(no_framestack_shape),
        replay_capacity=config.replay.capacity,
        action_shape=train_envs.act_space["action"].shape,
        action_dtype=train_envs.act_space["action"].dtype,
        extra_storage_types=[common.ReplayElement("is_first", (), np.int32),
                             common.ReplayElement("is_last", (), np.int32)]
    )

    eval_replay = buffer_cls(
        stack_size=config.framestack,
        n_envs=num_eval_envs,
        batch_size=config.dataset.batch,
        subseq_len=config.dataset.length,
        persistent=config.replay.persistent,
        seed=config.seed,
        observation_shape=train_envs.obs_space["image"].shape,
        replay_capacity=config.replay.capacity//100,
        action_shape=train_envs.act_space["action"].shape,
        action_dtype=train_envs.act_space["action"].dtype,
        extra_storage_types=[common.ReplayElement("is_first", (), np.int32),
                             common.ReplayElement("is_last", (), np.int32)]
    )

    train_add = lambda obs: train_replay.add(obs["image"], obs["action"],
                                             obs["reward"], obs["is_terminal"],
                                             obs["is_first"],
                                             obs["is_last"],
                                             episode_end=obs["is_last"])
    eval_add = lambda obs: eval_replay.add(obs["image"], obs["action"],
                                           obs["reward"], obs["is_terminal"],
                                           obs["is_first"],
                                           obs["is_last"],
                                           episode_end=obs["is_last"])

    outputs = [
        common.TerminalOutput(),
        common.JSONLOutput(logdir),
        common.TensorBoardOutput(logdir),
    ]

    print('Create agent.')
    agnt = agent.Agent(config, obs_space, act_space, step)
    train_agent = common.CarryOverState(agnt.train, init_state=agnt.init_policy_state(1)[0])

    # logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    train_driver = common.Driver(train_envs, config.envs, init_state=agnt.init_policy_state(config.envs))
    train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
    train_driver.on_step(lambda tran: step.increment(config.envs))
    train_driver.on_step(train_add)
    train_driver.on_reset(train_add)
    eval_driver = common.Driver(eval_envs, num_eval_envs, init_state=agnt.init_policy_state(num_eval_envs))
    eval_driver.on_step(eval_add)
    eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))

    prefill = max(0, config.prefill - train_replay.total_steps)
    if prefill:
        print(f'Prefill dataset ({prefill} steps).')
        random_agent = common.RandomAgent(act_space, 42)
        train_driver(random_agent, steps=prefill, episodes=1)
        eval_driver(random_agent, episodes=1)
        train_driver.reset()
        eval_driver.reset()

    def postprocess_sample(sample_dict):
        # Need to transpose last 2 channels to make sure colors are continuous
        image = sample_dict["state"].transpose(0,1,2,3,5,4)
        sample_dict["image"] = image.reshape(*image.shape[:-2], -1)
        sample_dict['same_trajectory'] = sample_dict['same_trajectory'].astype(np.int32)
        sample_dict = {k: train_driver._convert(jnp.array(v)) for k, v in sample_dict.items()}
        sample_dict = {k: jnp.array(v) for k, v in sample_dict.items()}
        sample_dict["is_terminal"] = sample_dict["terminal"]
        return sample_dict['image'], sample_dict['action'], sample_dict['reward'], sample_dict['is_first'], sample_dict["terminal"]

    print('Initialize agent.')
    sample = train_replay.sample(batch_size=1)[1]
    obs, action, reward, is_first, terminal = postprocess_sample(sample)
    train_agent(obs, action, reward, is_first, terminal)

    train_agent = common.CarryOverState(agnt.train, init_state=agnt.init_policy_state(config.dataset.batch)[0])
    
    print('Pretrain agent.')
    for i in range(config.pretrain):
        start = time.time()
        sample = train_replay.sample(batch_size=config.dataset.batch)[1]
        sample_time = time.time() - start
        obs, action, reward, is_first, terminal = postprocess_sample(sample)
        start = time.time()
        metrics, rec_obs = train_agent(obs, action, reward, is_first, terminal)
        train_time = time.time() - start
        metrics["sample_time"] = sample_time
        metrics["train_time"] = train_time
        if should_log(i):
            wandb.log(metrics)
            print(metrics)

    def train_policy(sample_dict, state, reset):
        return agnt.policy(obs=jnp.expand_dims(sample_dict['image'],1), is_first=sample_dict['is_first'], state=state, reward=sample_dict['reward'],
                                            mode='explore'
                                            if should_expl(step) else 'train',
                        reset=reset)
    eval_policy = lambda sample_dict, state, reset: agnt.policy(obs=jnp.expand_dims(sample_dict['image'],1),
                                                                is_first=sample_dict['is_first'],
                                                                state=state,
                                                                reward=sample_dict['reward'],
                                                                mode='eval',
                                                                reset=reset)

    def train_step(_):
        x = should_train(step)
        for _ in range(x):
            for _ in range(config.train_steps):
                start = time.time()
                sample = train_replay.sample()[1]
                sample_time = time.time() - start
                obs, action, reward, is_first, terminal = postprocess_sample(sample)
                
                start = time.time()
                metrics, rec_obs = train_agent(obs, action, reward, is_first, terminal)
                metrics["train_time"] = time.time() - start
                metrics["sample_time"] = sample_time
                
        if should_log(step):
            # Log steps
            metrics["environment_steps"] = step.value
            print(metrics)
            wandb.log(metrics)

    train_driver.on_step(train_step)

    while step < config.steps:
        # logger.write()
        print('Start evaluation.')
        # logger.add(agnt.report(next(eval_dataset)), prefix='eval')
        eval_driver(eval_policy, episodes=config.eval_eps)
        print('Start training.')
        train_driver(train_policy, steps=config.eval_every)
        # agnt.save(logdir / 'variables.pkl')
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == '__main__':
    step = common.Counter(0)
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
    parsed, remaining = common.Flags(configs=['defaults']).parse(
        known_only=True)
    config = common.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = common.Flags(config).parse(remaining)

    group_name = "%s" % (config.task)
    name = "%s_%s_%d" % (
        config.wandb_run_id, config.task, np.random.randint(100000000))

    wandb.init(project=config.wandb_project,
               entity=config.wandb_entity,
               config=config,
               group=group_name,
               name=name,
               sync_tensorboard=False,
               mode=config.wandb_mode)
    if config.profile:
        fn = os.path.join(pathlib.Path(config.logdir).expanduser(),'profile.prof')
        cProfile.run('main(config)', filename=fn)
    else:
        main(config)