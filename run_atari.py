import argparse
import os.path as osp
from gym.wrappers import Monitor
import numpy as np

from common import logger
from common.atari_wrappers import make_atari, wrap_deepmind
import dqfd


def train(args):
    total_timesteps = int(args.num_timesteps)
    pre_train_timesteps = int(args.pre_train_timesteps)
    seed = args.seed

    env = make_env(args.env, args.seed, args.max_episode_steps, wrapper_kwargs={'frame_stack': True, 'episode_life': True})
    if args.save_video_interval != 0:
        env = Monitor(env, osp.join(logger.get_dir(), "videos"), video_callable=(lambda ep: ep % args.save_video_interval == 0), force=True)
    model = dqfd.learn(
        env=env,
        network='cnn',
        checkpoint_path=args.save_path,
        seed=seed,
        total_timesteps=total_timesteps,
        pre_train_timesteps=pre_train_timesteps,
        load_path=args.load_path,
        demo_path=args.demo_path,
        buffer_size=int(args.buffer_size),
        batch_size=args.batch_size,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        epsilon_schedule=args.epsilon_schedule,
        lr=args.lr)

    return model, env


def make_env(env_id, seed=None, max_episode_steps=None, wrapper_kwargs=None):
    wrapper_kwargs = wrapper_kwargs or {}
    env = make_atari(env_id, max_episode_steps)
    env.seed(seed)
    env = wrap_deepmind(env, **wrapper_kwargs)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str, default='atari')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num_timesteps', help='', type=float, default=2e6)
    parser.add_argument('--pre_train_timesteps', help='', type=float, default=100000)
    parser.add_argument('--max_episode_steps', help='', type=int, default=10000)
    parser.add_argument('--network', help='', type=str, default='cnn')
    parser.add_argument('--save_path', help='Path to save trained model to', default='data/temp', type=str)
    parser.add_argument('--load_path', help='Path to load trained model to', default='data/temp', type=str)
    parser.add_argument('--save_video_interval', help='Save video every x episodes (0 = disabled)', default=10, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 2000', default=2000, type=int)
    parser.add_argument('--demo_path', help='Directory to save learning curve data.', default="data/demo/human.BreakoutNoFrameskip-v4.pkl", type=str)
    parser.add_argument('--log_path', help='Path to save log to', default='data/logs', type=str)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--batch_size', help='batch size for both pretraining and training', type=int, default=64)
    parser.add_argument('--buffer_size', help='experience replay buffer size', type=float, default=5e5)
    parser.add_argument('--exploration_fraction', help='anneal exploration epsilon for this fraction of total training steps', type=float, default=0.1)
    parser.add_argument('--exploration_final_eps', help='exploration epsilon after annealing', type=float, default=0.1)
    parser.add_argument('--epsilon_schedule', help='linear or constant', type=str, default='linear')
    parser.add_argument('--lr', help='learning rate', type=float, default=5e-4)
    args = parser.parse_args()

    logger.configure(args.log_path)
    model, env = train(args)

    if args.play:
        logger.log("Running trained model")
        obs = env.reset()
        obs = np.expand_dims(np.array(obs), axis=0)

        state = model.initial_state if hasattr(model, 'initial_state') else None

        episode_rew = np.zeros(1)
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs)
            else:
              actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions.numpy())
            obs = np.expand_dims(np.array(obs), axis=0)
            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0
                    env.reset()
    env.close()

    return model


if __name__ == "__main__":
    main()
