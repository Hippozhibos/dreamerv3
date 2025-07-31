### 2025-06-19
- demo:'python dreamerv3/main.py   --logdir ~/logdir/dreamer/{timestamp}   --configs dmc_proprio  --run.train_ratio 32'
- BUG: 'mujoco.FatalError: gladLoadGL error'
- 已经DEBUG，但暂时没法启动Tensorboard等可视化界面；

### 2025-06-19
- 启动命令:'python CyberMice/main.py   --logdir ~/logdir/dreamer/{timestamp}   --configs dmc_proprio  --run.train_ratio 32'

### 2025-07-31
- 训练的查看方式看readme，用scope
- dmc_proprio可以正常启动，看看数据流是怎么回事
  - [TODO] 将mice pusher、mice go2环境连接上去