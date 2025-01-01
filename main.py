# Import necessary libraries
import argparse
import os
import pprint
import torch
import numpy as np
from datetime import datetime
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer, Batch
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.trainer import offpolicy_trainer
from torch.distributions import Independent, Normal
from tianshou.exploration import GaussianNoise
from tianshou.utils import BasicLogger
from env import make_aigc_env
# from env import make_aigc_env
from env.routeEnv import make_route_env
from policy import DiffusionOPT
from diffusion import Diffusion
from diffusion.model import MLP, DoubleCritic
import warnings
from tianshou.utils import TensorboardLogger
import os
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# Ignore warnings
warnings.filterwarnings('ignore')
from tianshou.utils import BasicLogger


class CustomLogger(TensorboardLogger):
    def __init__(self, writer, log_dir="./logs", action_file="actions.json"):
        super().__init__(writer)
        self.action_log = []  # 用于存储所有动作
        self.log_dir = log_dir  # 日志文件目录
        self.action_file = action_file  # 动作保存文件名

        # 创建目录
        os.makedirs(self.log_dir, exist_ok=True)

    def log_train_data(self, collect_result, step):
        super().log_train_data(collect_result, step)

        # 记录训练时的动作
        if "acts" in collect_result:  # 确保动作信息可用
            actions = collect_result["acts"]
            self.action_log.extend(actions)  # 将动作存储到日志中

            # 示例：记录动作分布到 TensorBoard
            unique_actions, counts = np.unique(actions, return_counts=True)
            for action, count in zip(unique_actions, counts):
                self.writer.add_scalar(f"train/action_{action}", count, step)

    def save_actions(self, file_format="json"):
        """保存动作到文件."""
        file_path = os.path.join(self.log_dir, self.action_file)
        if file_format == "json":
            with open(file_path, "w") as f:
                json.dump(self.action_log, f)
            print(f"Actions saved to {file_path} (JSON format)")
        else:
            raise ValueError("Unsupported file format. Use 'json'.")

# class CustomLogger(BasicLogger):
#
#     def log_train_data(self, collect_result, step):
#         super().log_train_data(collect_result, step)
#         self.writer.add_scalar("train/it_per_s", collect_result["speed"], step)
#         self.writer.add_scalar("train/reward", collect_result["rews"].mean(), step)

# Define a function to get command line arguments
def get_args():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration-noise", type=float, default=0.1)  # Noise added to actions for exploration
    parser.add_argument('--algorithm', type=str, default='diffusion_opt')  # Algorithm to use
    parser.add_argument('--seed', type=int, default=1)  # Random seed for reproducibility
    parser.add_argument('--buffer-size', type=int, default=5e6)  # Size of the replay buffer
    parser.add_argument('-e', '--epoch', type=int, default=100)  # Number of epochs to train
    parser.add_argument('--step-per-epoch', type=int, default=1)  # Number of steps per epoch
    parser.add_argument('--step-per-collect', type=int, default=1)  # Number of steps per data collection
    parser.add_argument('-b', '--batch-size', type=int, default=512)  # Batch size for training
    parser.add_argument('--wd', type=float, default=1e-4)  # Weight decay for optimizer
    parser.add_argument('--gamma', type=float, default=0.98)  # Discount factor for reward
    parser.add_argument('--n-step', type=int, default=3)  # Number of steps for n-step return
    parser.add_argument('--training-num', type=int, default=10)  # Number of training environments
    parser.add_argument('--test-num', type=int, default=1)  # Number of test environments
    parser.add_argument('--logdir', type=str, default='log')  # Directory to save logs
    parser.add_argument('--log-prefix', type=str, default='default')  # Prefix for log files
    parser.add_argument('--render', type=float, default=0.1)  # Render the environment with this frequency
    parser.add_argument('--rew-norm', type=int, default=0)  # Normalize rewards
    parser.add_argument('--device', type=str, default='cuda:1')  # Device to use for training (CPU or GPU)
    parser.add_argument('--resume-path', type=str, default=None)  # Path to resume training from a checkpoint
    parser.add_argument('--watch', action='store_true', default=False)  # Watch the performance of the trained model
    parser.add_argument('--lr-decay', action='store_true', default=False)  # Apply learning rate decay
    parser.add_argument('--note', type=str, default='')  # Additional notes

    # for diffusion
    parser.add_argument('--actor-lr', type=float, default=2e-4)  # Learning rate for the actor
    parser.add_argument('--critic-lr', type=float, default=2e-4)  # Learning rate for the critic
    parser.add_argument('--tau', type=float, default=0.005)  # Soft update parameter for target networks
    # adjust
    parser.add_argument('-t', '--n-timesteps', type=int, default=6)  # Number of timesteps for diffusion chain
    parser.add_argument('--beta-schedule', type=str, default='vp',
                        choices=['linear', 'cosine', 'vp'])  # Schedule for beta in diffusion

    # With Expert: bc-coef True
    # Without Expert: bc-coef False
    # parser.add_argument('--bc-coef', default=False) # Apr-04-132705
    parser.add_argument('--bc-coef', default=False)

    # for prioritized experience replay
    parser.add_argument('--prioritized-replay', action='store_true', default=False)
    parser.add_argument('--prior-alpha', type=float, default=0.4)#
    parser.add_argument('--prior-beta', type=float, default=0.4)#

    # Parse arguments and return them
    args = parser.parse_known_args()[0]
    return args

#
def main(args=get_args()):
    # create environments
    env, train_envs, test_envs = make_route_env(args.training_num, args.test_num)
    args.state_shape = env.observation_space.shape[0]
    # args.action_shape = env._get_current_observation().shape
    args.action_shape = env.action_space.n
    args.max_action = 1.

    args.exploration_noise = args.exploration_noise * args.max_action
    # seed
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    # create actor
    actor_net = MLP(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    )
    # Actor is a Diffusion model
    actor = Diffusion(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        model=actor_net,
        max_action=args.max_action,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.n_timesteps,
        bc_coef = args.bc_coef
    ).to(args.device)
    actor_optim = torch.optim.AdamW(
        actor.parameters(),
        lr=args.actor_lr,
        weight_decay=args.wd
    )
    # 创建一个AdamW优化器，用于更新actor网络的参数。
    # AdamW是一种变种的Adam优化器，具有权重衰减（weight decay）功能，
    # 可以帮助防止过拟合。优化器的学习率（lr）和权重衰减系数（weight decay）
    # 分别由命令行参数args.actor_lr和args.wd指定。
    # Create critic
    critic = DoubleCritic(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    ).to(args.device)
    critic_optim = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )

    ## Setup logging
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    log_path = os.path.join(args.logdir, args.log_prefix, "diffusion", time_now)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    # logger = TensorboardLogger(writer)
    logger = CustomLogger(writer, log_dir=str(log_path), action_file="actions.json")
    # def dist(*logits):
    #    return Independent(Normal(*logits), 1)

    # Define policy
    policy = DiffusionOPT(
        args.state_shape,
        actor,
        actor_optim,
        args.action_shape,
        critic,
        critic_optim,
        # dist,
        args.device,
        tau=args.tau,
        gamma=args.gamma,
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.epoch,
        bc_coef=args.bc_coef,
        action_space=env.action_space,
        exploration_noise = args.exploration_noise,
    )

    # Load a previous policy if a path is provided
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # Setup buffer
    if args.prioritized_replay:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.prior_alpha,
            beta=args.prior_beta,
        )
    else:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs)
        )

    # Setup collector
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # Trainer
    if not args.watch:
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False
        )
        pprint.pprint(result)

    # Watch the performance
    # python main.py --watch --resume-path log/default/diffusion/Jul10-142653/policy.pth
    if __name__ == '__main__':
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1) #, render=args.render
        print(result)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
        # 获取每个回合的动作
        state = env.reset()
        state_batch = Batch(obs=np.array([state]), info={})
        # 使用策略进行预测（在评估模式下）
        with torch.no_grad():  # 关闭梯度计算，加速推理
            for _ in range(100):
                acts = policy(state_batch).act[0]  # 获取动作
                acts = acts.cpu().numpy()
                action = np.round(((acts + 1) / 2) * 4)
                env._num_steps =0
                env.step(action)
                rewards, noise = env.render()
                print("Predicted action:", action, "Predicted rewards:", rewards,"Predicted action:", noise)
        # actions = result['acts']  # 返回的是一个列表，按回合存储所有动作

        # 输出动作
        # for i, episode_actions in enumerate(actions):
        #     print(f"Episode {i + 1} actions: {episode_actions}")
        # collector.collect(n_episode=10, render=1 / 35)

        # Output actions
        # for i in range(len(result["acts"])):
        #     print(f"Action {i}: {result['acts'][i]}")

if __name__ == '__main__':
    # for i in range(20):
    main(get_args())
