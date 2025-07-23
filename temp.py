import glob
import os
import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# # Save a checkpoint every 1000 steps
# checkpoint_callback = CheckpointCallback(
#     save_freq=100,
#     save_path="log/ttt",
#     name_prefix="model",
#     save_replay_buffer=True,
#     save_vecnormalize=True,
# )
#
# model = SAC("MlpPolicy", "Pendulum-v1")
# model.learn(
#     1000,
#     callback=checkpoint_callback,
#     log_interval=1,
#     tb_log_name="cnm",
#     reset_num_timesteps=False,
#     progress_bar=True
# )

# # 动态获取所有保存的模型文件
# save_path = f"log/ttt"
# model_files = glob.glob(os.path.join(save_path, "model_*.zip"))
# model_files.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))  # 按步数排序
#
# # 取最后10个模型
# last_n_models = model_files[-10:]
#
# models = [
#     {
#         "name": f"{file.split('.')[0].split('/')[-1]}",
#         "path": f"log/ttt"
#     } for file in last_n_models
# ]
#
# print(models)

print(f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
