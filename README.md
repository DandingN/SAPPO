# SAPPO
Spatiotemporal-Aware Reinforcement Learning for Maritime Search Path Planning

本人论文《融合时空感知的强化学习海上搜索路径规划》源码，2025年6月30日已投稿

引用请联系`pc_yang18@163.com`

# 核心工作
1. 使用多维高斯概率分布拟合遇险目标的分布概率，`data`文件夹为包含1000个场景的标准测试集
2. 标准测试集通过将问题建模为混合整数线性规划问题，通过`Gurobi`求解器求解出最优解，`data`中的`object`为最优解的90%
3. 核心强化学习算法基于`stable-baselines3-contrib`编写，在PPO算法基础上，融合LSTM模块增强时空特征感知能力
4. 提出基于阈值的场景切换机制，在单个场景的某回合POS达到场景对应的`object`值时切换至下一个场景
5. 通过在900个训练集上进行训练，并在剩余100个测试集上进行测试，本文成功验证了模型的泛化性，在测试集上的效果均达到了最优解的90%以上

# 突出贡献
在海上搜救领域，尤其是海上 **搜救/搜索** 路径规划领域的强化学习研究中，止步于个别固定场景的训练及测试，不具备泛化能力，与启发式算法相比，丧失竞争优势。

本文通过上述工作，大幅提升泛化能力，并进行了充分验证

# 特别感谢 stable-baselines3-contrib
https://github.com/Stable-Baselines-Team/stable-baselines3-contrib