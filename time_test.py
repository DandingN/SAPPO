#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/5/4 下午8:17
# @function: the script is used to do something.
# @version : V1
# 动态获取所有保存的模型文件
import glob
import os
from test_model import test_model
import time
import datetime

def run_test():
    model_name = 20250504_12
    save_path = f"models/{model_name}"
    model_files = glob.glob(os.path.join(save_path, "model_*.zip"))
    model_files.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))  # 按步数排序

    # 取最后10个模型
    last_n_models = model_files[5:15]

    models = [
        {
            "name": f"{file.split('.')[0].split('/')[-1]}",
            "path": f"models/{model_name}/",
            "model_name": model_name
        } for file in last_n_models
    ]

    test_model(models)


# 计算当前时间到明天凌晨 3:00 的等待时间
def wait_until_3am():
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    target_time = datetime.datetime(year=tomorrow.year, month=tomorrow.month, day=tomorrow.day, hour=4, minute=0, second=0)
    wait_seconds = (target_time - now).total_seconds()
    print(f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"等待到明天凌晨 3:30，剩余时间: {wait_seconds} 秒")
    time.sleep(wait_seconds)
    run_test()


if __name__ == "__main__":
    # while True:
        # wait_until_3am()
        run_test()
