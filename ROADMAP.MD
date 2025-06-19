### 2025-06-19
- demo:'python dreamerv3/main.py   --logdir ~/logdir/dreamer/{timestamp}   --configs dmc_proprio  --run.train_ratio 32'
- BUG: 'mujoco.FatalError: gladLoadGL error'
    - 'libGL error: MESA-LOADER: failed to open swrast: /home/zhangzhibo/anaconda3/envs/mujoco/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1) (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
/home/zhangzhibo/anaconda3/envs/mujoco/lib/python3.11/site-packages/glfw/__init__.py:914: GLFWError: (65543) b'GLX: Failed to create context: BadValue (integer parameter out of range for operation)'
  warnings.warn(message, GLFWError)
Traceback (most recent call last):
  File "/home/zhangzhibo/dreamerv3/dreamerv3/main.py", line 276, in <module>
    main()
  File "/home/zhangzhibo/dreamerv3/dreamerv3/main.py", line 69, in main
    embodied.run.train(
  File "/home/zhangzhibo/dreamerv3/embodied/run/train.py", line 97, in train
    driver(policy, steps=10)
  File "/home/zhangzhibo/dreamerv3/embodied/core/driver.py", line 54, in __call__
    step, episode = self._step(policy, step, episode)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zhangzhibo/dreamerv3/embodied/core/driver.py", line 65, in _step
    obs = [env.step(act) for env, act in zip(self.envs, acts)]
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zhangzhibo/dreamerv3/embodied/core/driver.py", line 65, in <listcomp>
    obs = [env.step(act) for env, act in zip(self.envs, acts)]
           ^^^^^^^^^^^^^
  File "/home/zhangzhibo/dreamerv3/embodied/core/wrappers.py", line 86, in step
    return self.env.step({**action, self._key: clipped})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zhangzhibo/dreamerv3/embodied/core/wrappers.py", line 254, in step
    obs = self.env.step(action)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/home/zhangzhibo/dreamerv3/embodied/core/wrappers.py", line 223, in step
    obs = self.env.step(action)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/home/zhangzhibo/dreamerv3/embodied/core/wrappers.py", line 109, in step
    return self.env.step({**action, self._key: orig})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zhangzhibo/dreamerv3/embodied/envs/dmc.py", line 74, in step
    obs[key] = self._dmenv.physics.render(*self._size, camera_id=self._camera)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zhangzhibo/anaconda3/envs/mujoco/lib/python3.11/site-packages/dm_control/mujoco/engine.py", line 223, in render
    camera = Camera(
             ^^^^^^^
  File "/home/zhangzhibo/anaconda3/envs/mujoco/lib/python3.11/site-packages/dm_control/mujoco/engine.py", line 711, in __init__
    if self._physics.contexts.mujoco is not None:
       ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zhangzhibo/anaconda3/envs/mujoco/lib/python3.11/site-packages/dm_control/mujoco/engine.py", line 533, in contexts
    self._make_rendering_contexts()
  File "/home/zhangzhibo/anaconda3/envs/mujoco/lib/python3.11/site-packages/dm_control/mujoco/engine.py", line 519, in _make_rendering_contexts
    mujoco_context = wrapper.MjrContext(self.model, render_context)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zhangzhibo/anaconda3/envs/mujoco/lib/python3.11/site-packages/dm_control/mujoco/wrapper/core.py", line 608, in __init__
    ptr = ctx.call(mujoco.MjrContext, model.ptr, font_scale)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zhangzhibo/anaconda3/envs/mujoco/lib/python3.11/site-packages/dm_control/_render/executor/render_executor.py", line 138, in call
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
mujoco.FatalError: gladLoadGL error'