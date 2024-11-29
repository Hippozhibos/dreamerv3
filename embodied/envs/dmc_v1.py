import functools
import os

import embodied
import numpy as np

import imageio  # 用于保存视频


class DMC(embodied.Env):

  DEFAULT_CAMERAS = dict(
      quadruped=2,
      locom_rodent=4,
  )

  def __init__(
      self, env, repeat=1, size=(64, 64), image=True, camera=-1):
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
      # os.environ['MUJOCO_GL'] = 'MUJOCO_PY'
    if isinstance(env, str):
      domain, task = env.split('_', 1)
      if camera == -1:
        camera = self.DEFAULT_CAMERAS.get(domain, 0)
      if domain == 'cup':  # Only domain with multiple words.
        domain = 'ball_in_cup'
      if domain == 'manip':
        from dm_control import manipulation
        env = manipulation.load(task + '_vision')
      elif domain == 'locom':
        # camera 0: topdown map
        # camera 2: shoulder
        # camera 4: topdown tracking
        # camera 5: eyes
        from dm_control.locomotion.examples import basic_rodent_2020
        env = getattr(basic_rodent_2020, task)()
      else:
        from dm_control import suite
        env = suite.load(domain, task)
    self._dmenv = env
    from . import from_dm
    self._env = from_dm.FromDM(self._dmenv)
    self._env = embodied.wrappers.ExpandScalars(self._env)
    self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
    self._size = size
    self._image = image
    self._camera = camera
    self._frames = []  # 用于存储渲染帧

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    key = 'image' if self._image else 'log_image'
    spaces[key] = embodied.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    # 检查动作空间是否有效
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    
    # 执行环境的一步
    obs = self._env.step(action)
    
    # 渲染并保存当前帧
    frame = self._dmenv.physics.render(*self._size, camera_id=self._camera)
    self._frames.append(frame)  # 将帧添加到帧列表

    key = 'image' if self._image else 'log_image'
    obs[key] = self._dmenv.physics.render(*self._size, camera_id=self._camera)
    
    # 确保观察结果中的值是有限的
    for key, space in self.obs_space.items():
      if np.issubdtype(space.dtype, np.floating):
        assert np.isfinite(obs[key]).all(), (key, obs[key])
    return obs
  
  def reset(self):
        """重置环境并清空帧列表。"""
        self._frames = []  # 清空之前存储的帧
        return self._env.reset()

  def save_video(self, filename="rendered_video.mp4", fps=30):
      """将收集的渲染帧保存为视频文件。"""
      if not self._frames:
          print("No frames to save. Run the environment first.")
          return
      imageio.mimsave(filename, self._frames, fps=fps)
      print(f"Video saved to {filename}")