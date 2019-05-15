import copy
import glob
import os
import time
import types
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
import pickle
import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage

from gatedpixelcnn_bonus import PixelBonus
from skimage.transform import resize

#from visualize import visdom_plot

def update_tf_wrapper_args(args, tf_flags):
    """
    take input command line args to DQN agent and update tensorflow wrapper default
    settings
    :param args:
    :param FLAGS:
    :return:
    """
    # doesn't support boolean arguments
    to_parse = args.wrapper_args
    if to_parse:
        for kwarg in to_parse:
            keyname, val = kwarg.split('=')
            if keyname in ['ckpt_path', 'data_path', 'samples_path', 'summary_path']:
                # if directories don't exist, make them
                if not os.path.exists(val):
                    os.makedirs(val)
                tf_flags.update(keyname, val)
            elif keyname in ['data', 'model']:
                tf_flags.update(keyname, val)
            elif keyname in ['mmc_beta']:
                tf_flags.update(keyname, float(val))
            else:
                tf_flags.update(keyname, int(val))
    return tf_flags

class DotDict(object):
    def __init__(self, dict):
        self.dict = dict

    def __getattr__(self, name):
        return self.dict[name]

    def update(self, name, val):
        self.dict[name] = val

    # can delete this later
    def get(self, name):
        return self.dict[name]

FLAGS = DotDict({
    'img_height': 42,
    'img_width': 42,
    'channel': 1,
    'data': 'mnist',
    'conditional': False,
    'num_classes': None,
    'filter_size': 3,
    'init_fs': 7,
    'f_map': 16,
    'f_map_fc': 16,
    'colors': 8,
    'parallel_workers': 1,
    'layers': 3,
    'epochs': 25,
    'batch_size': 16,
    'model': '',
    'data_path': 'data',
    'ckpt_path': 'ckpts',
    'samples_path': 'samples',
    'summary_path': 'logs',
    'restore': True,
    'nr_resnet': 1,
    'nr_filters': 32,
    'nr_logistic_mix': 5,
    'resnet_nonlinearity': 'concat_elu',
    'lr_decay': 0.999995,
    'lr': 0.00005,
    'num_ds': 1
})

imresize = resize


args = get_args()
useNeural = bool(args.useNeural)

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    """
    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None
    """

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.useNeural:
        #FLAGS = update_tf_wrapper_args(args,)
        tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        pixel_bonus = PixelBonus(FLAGS, sess)
        tf.initialize_all_variables().run(session=sess)
        if args.loadNeural is not None:
            pixel_bonus.load_model(args.loadNeural)

        #with tf.variable_scope('step'):
        #    self.step_op = tf.Variable(0, trainable=False, name='step')
        #    self.step_input = tf.placeholder('int32', None, name='step_input')
        #    self.step_assign_op = self.step_op.assign(self.step_input)


    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)


    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)

    steper =0
    img_scale = 1
    psc_weight = float(args.pscWeight)
    psc_rollout=list()

    start = time.time()
    for j in range(num_updates):
        step_counter = 0
        psc_tot=list()
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            psc_add = 0
            if args.useNeural:
                step_batch = 0
                for i in obs[0]:
                    frame = imresize((i / img_scale).cpu().numpy(), (42, 42), order=1)
                    psc = pixel_bonus.bonus(i, steper)
                    psc_add += psc
                    reward[step_batch] += torch.FloatTensor([psc])
                    steper += 1
                    step_batch += 1

                psc_add = psc_add / step_batch
            else:
                psc_add = 0


            """
            for info in infos:
                if 'episode' in info.keys():
                    print(reward)
                    episode_rewards.append(info['episode']['r'])
            """

            # FIXME: works only for environments with sparse rewards
            for idx, eps_done in enumerate(done):
                if eps_done:
                    episode_rewards.append(reward[idx])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            print('Saving model')
            print()

            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model, hasattr(envs.venv, 'ob_rms') and envs.venv.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success rate {:.2f}\n".
                format(
                    j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    np.count_nonzero(np.greater(episode_rewards, 0)) / len(episode_rewards)
                )
            )

        if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0:
            eval_envs = make_vec_envs(args.env_name, args.seed + args.num_processes, args.num_processes,
                                args.gamma, eval_log_dir, args.add_timestep, device, True)

            if eval_envs.venv.__class__.__name__ == "VecNormalize":
                eval_envs.venv.ob_rms = envs.venv.ob_rms

                # An ugly hack to remove updates
                def _obfilt(self, obs):
                    if self.ob_rms:
                        obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                        return obs
                    else:
                        return obs

                eval_envs.venv._obfilt = types.MethodType(_obfilt, envs.venv)

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)
                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
                len(eval_episode_rewards),
                np.mean(eval_episode_rewards)
            ))
    if useNeural:
        pixel_bonus.save_model(str(args.nameDemonstrator) + "neural", step)

        """
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass
        """

if __name__ == "__main__":
    main()
