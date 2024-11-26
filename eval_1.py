import warnings
import numpy as np
import jax
from functools import partial as bind
from embodied.envs import dmc

import dreamerv3
import embodied

logdir = embodied.Path('~/logdir/20241117T135935-example')
config = embodied.Config.load(logdir / 'config.yaml')


env = dmc.DMC('locom_rodent_maze_forage', image=False, camera=5)
env = dreamerv3.wrap_env(env, config)

step = embodied.Counter()
agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
checkpoint = embodied.Checkpoint()
checkpoint.agent = agent
checkpoint.load(logdir / 'checkpoint.ckpt', keys=['agent'])

# Initialize state as a JAX device array
prevlat = jax.device_put(np.zeros((1,)))  # Adjust the shape and initial value as needed
prevact = jax.device_put(np.zeros((1,)))  # Adjust the shape and initial value as needed
state = (prevlat, prevact)

act = {'action': env.act_space['action'].sample(), 'reset': np.array(True)}
while True:
    obs = env.step(act)
    # obs = {k: v[None] for k, v in obs.items()}
    obs = {k: (np.array([v]) if np.isscalar(v) else v) for k, v in obs.items()}


    #  # Flatten the camera observation
    # if 'walker_egocentric_camera' in obs:
    #     obs['walker_egocentric_camera'] = obs['walker_egocentric_camera'].reshape(-1)
    # obs = {k: v.reshape(-1, 1) if v.ndim == 1 else v for k, v in obs.items()}

    # Flatten all observations
    # obs = {k: v.reshape(-1) for k, v in obs.items()}

    act, state = agent.policy(obs, state, mode='eval')
    act = {'action': act['action'][0], 'reset': obs['is_last'][0]}