import warnings
from functools import partial as bind
from embodied.envs import dmc

import dreamerv3
import embodied

logdir = embodied.Path('~/logdir/20241117T135935-example')
config = embodied.Config.load(logdir / 'config.yaml')


env = dmc.DMC('locom_rodent_maze_forage', image=False, camera=5)
env = dreamerv3.wrap_env(env, config)

step = embodied.Counter()
agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
checkpoint = embodied.Checkpoint()
checkpoint.agent = agent
checkpoint.load(logdir / 'checkpoint.ckpt', keys=['agent'])

state = None
act = {'action': env.act_space['action'].sample(), 'reset': np.array(True)}
while True:
    obs = env.step(act)
    obs = {k: v[None] for k, v in obs.items()}
    act, state = agent.policy(obs, state, mode='eval')
    act = {'action': act['action'][0], 'reset': obs['is_last'][0]}