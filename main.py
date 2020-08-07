import argparse
from collections import deque

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import autograd.numpy as np
from scipy.stats import entropy

import models
import utils
from utils import one_hot_ify
import rl_tools

def single_run(
        args,
        logger,
        env,
        eval_env,
        num_episodes = 100,
        num_eval_episodes = 25,
        max_steps = 200,
        γ = 1.0,
        lr_actor = 0.01,
        lr_critic = 0.05,
        pol_ent=1):
    """Main algo to train an RL agent"""
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    return_run = np.zeros(num_episodes)
    samples_run = np.zeros(num_episodes)
    actor = models.SigmoidPolicy(num_states, num_actions)
    actor_opt = optim.Adam(actor.parameters(), lr=lr_actor)
    critic = models.Critic(num_states, num_actions)
    critic_opt = optim.Adam(critic.parameters(), lr=lr_critic)

    actor_params_sizes = torch.tensor(np.cumsum([0] + [len(t.flatten()) for t in list(actor.parameters())]))
    gradient_network = models.GradientNetwork(num_states, actor_params_sizes[-1])
    gradient_network_opt = optim.Adam(gradient_network.parameters(), lr=lr_critic)

    evaluations = []
    # bar = pyprind.ProgBar(num_episodes)
    for episode in range(num_episodes):
        print ("episode", episode)
        # bar.update()
        obs = env.reset()
        obs_hist = deque()
        log_prob_a_hist = deque()
        adv_hist = deque()
        q_sa_target_hist = deque()
        q_sa_hist = deque()
        return_all_eval_episodes = np.zeros(num_eval_episodes)
        total_vae_loss = 0
        scale_gamma = 1.0
        actor_params_list = []

        gradient_td_error_loss = deque()

        # Detached params and pointers
        actor_params = (torch.cat([t.flatten() \
                for t in list(actor.parameters())]).view(1,-1)).clone().detach().requires_grad_(True)
        actor_params_list = list(actor.parameters())

        # *** COLLECT DATA ***
        for step in range(max_steps):

            # Predict gradient
            grad_output_current_state = gradient_network(one_hot_ify(obs, num_states), actor_params)

            # Get actor and critic values
            prob_a = actor(one_hot_ify(obs, num_states))
            q_s = critic(one_hot_ify(obs, num_states))
            a_dist = torch.distributions.Categorical(probs = prob_a)
            action = int(a_dist.sample().numpy()[0])

            # Log: action prob, advantage, q values
            log_prob_a_hist.append((a_dist.log_prob(torch.tensor(action))).view(1,-1))
            adv_hist.append((q_s.data[0,action] - (q_s.data[0,:]*prob_a.data[0,:]).sum()).view(1,-1))
            q_sa_hist.append((q_s[0,action]).view(1,-1))

            obs, rew, done, _ = env.step(action)
            obs_hist.append(obs)

            rew = rew + pol_ent * entropy(prob_a.data[0]) #added policy entropy to the reward function

            # Get log_prob with grad function, for gradient network
            log_prob = a_dist.log_prob(torch.tensor(action))
            with torch.no_grad():
                # Next actor critic values
                q_s_next = critic(one_hot_ify(obs, num_states))
                prob_a_next = actor(one_hot_ify(obs, num_states))
                v_next = (q_s_next*prob_a_next).sum()
                q_target = rew + γ*v_next
                q_sa_target_hist.append((q_target).view(1,-1))

                # Predict next gradient
                # TODO: experiment with conditioning on either logits or params
                # Also, since we are taking the params, we cannot do a max over
                # actions for the next grad state
                grad_output_next_state = gradient_network(one_hot_ify(obs,
                                                    num_states), actor_params)

                # Compute next gradient target
                # gradient_reward = a_dist.log_prob(torch.tensor(action)) * (q_s.data[0,action] - (q_s.data[0,:]*prob_a.data[0,:]).sum())
                adv = (q_s.data[0,action] -(q_s.data[0,:]*prob_a.data[0,:]).sum())
                gradient_reward = torch.autograd.grad(log_prob, actor_params_list, retain_graph=True)
                gradient_reward = (torch.cat([t.flatten() \
                        for t in list(gradient_reward)]).view(1,-1))
                gradient_target = gradient_reward*adv + γ * grad_output_next_state

            gradient_td_error = nn.MSELoss()(gradient_target, grad_output_current_state)
            gradient_td_error_loss.append(gradient_td_error.view(1,-1))

            samples_run[episode] = step+1

            if done:
                break

        # *** POLICY UPDATE ***

        critic_loss = nn.MSELoss()(torch.cat(list(q_sa_hist)), torch.cat(list(q_sa_target_hist)))
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        actor_opt.zero_grad()

        # Update with the pg_bell function
        if args.pg_bellman :
            gradient_network_opt.zero_grad()
            gradient_loss = torch.cat(list(gradient_td_error_loss)).sum()
            gradient_loss.backward()
            gradient_network_opt.step()
            # for t_index, t in enumerate(list(actor.parameters())):
            #     t.grad = - lamda_ent * (actor_params.grad[0, actor_params_sizes[t_index]:actor_params_sizes[t_index+1]]).view(t.shape)

            # for t_index, t in enumerate(list(actor.parameters())):
                # t.grad = - lamda_ent * (actor_params.grad[0, actor_params_sizes[t_index]:actor_params_sizes[t_index+1]]).view(t.shape)

            # Policy param Update
            # TODO: Now that the Gradient Network is updated, we should loop
            # through the state/action history and update the policy params?

            # Loop state/action history and collect grads
            grads = torch.zeros_like(grad_output_current_state)
            for obs in obs_hist:
                grads += gradient_network(one_hot_ify(obs, num_states), actor_params)
            grads = grads / len(obs_hist)
            grads = grads.flatten()
            start = 0
            # Grab new param grads, reshape to same size
            for p in actor_params_list:
                stop = start + p.nelement()
                g = grads[start:stop].view(p.size())
                p.grad = -g.clone() # clone otherwise opt won't work
                start=stop
            actor_opt.step()

        # If using standard ac policy gradient:
        else:
            actor_loss = -(torch.cat(list(log_prob_a_hist))*torch.cat(list(adv_hist))).sum()
            actor_loss.backward()
            actor_opt.step()


        # *** EVALUATION ***
        for eval_episode in range(num_eval_episodes):
            eval_obs = eval_env.reset()
            return_eval_episode = 0
            scale = 1.0
            for eval_step in range(max_steps):
                with torch.no_grad():
                    eval_prob_a = actor(one_hot_ify(eval_obs, num_states))
                    eval_a = torch.distributions.Categorical(probs = eval_prob_a).sample().numpy()[0]
                eval_obs, eval_rew, eval_done, _ = eval_env.step(eval_a)
                return_eval_episode += scale*eval_rew
                scale *= γ
                if eval_done:
                    break
            return_all_eval_episodes[eval_episode] = return_eval_episode
        return_run[episode] = np.mean(return_all_eval_episodes)
        print ("EvalRewards : ", episode, ":", np.mean(return_all_eval_episodes))
        evaluations.append(np.mean(return_all_eval_episodes))
        logger.record_reward(evaluations)

    logger.save()
    return return_run, samples_run, actor, critic

def main():
    """Load all settings and launch training"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, help='Environment name', default='GridWorld',
        choices=['GridWorld', 'WindyGridWorld', 'FrozenLake', 'FrozenLake8', 'Taxi', 'CliffWalking','twostateMDP'])
    parser.add_argument('--num_runs', type=int, help='Number of independent runs', default=1)
    parser.add_argument('--lr_actor', type=float, help='Learning rate for actor', default = 0.01)
    parser.add_argument('--lr_critic', type=float, help='Learning rate for critic', default = 0.05)
    parser.add_argument('--num_episodes', type=int, help='Number of episodes', default = 3000)
    # parser.add_argument('--lamda_ent', type=float, help='weigting for exploration bonus', default = 1.0)
    parser.add_argument("--use_logger", action="store_true", default=False, help='whether to use logging or not')
    parser.add_argument("--folder", type=str, default='./results/')
    # parser.add_argument('--num_latents', type=float, help='latent dimensions', default = 64)
    parser.add_argument('--seed', type=int, help='Random seed', default=0)
    parser.add_argument('--pol_ent', type=float, help='weigting for exploration bonus for the policy', default = 1.0)
    parser.add_argument("--pg_bellman", action="store_true", default=False, help='whether to use meta Bellman update')

    args = parser.parse_args()
    args.use_logger = True

    if args.pg_bellman :
        policy_name = "pg_bellman"
    else:
        policy_name = "baseline"

    if args.use_logger:
        logger = utils.Logger(experiment_name = policy_name, environment_name = args.env, folder = args.folder)
        logger.save_args(args)
        print ('Saving to', logger.save_folder)

    # Get environment
    env, eval_env, mdp = utils.get_env(args)

    # Set deterministic
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    env.reset()

    #####################
    # EXACT SOLUTION
    #####################
    rl_tools.exact_solution(args, mdp)

    #####################
    # MAIN TRAINING LOOP
    #####################
    print("Running")
    res_PGT = []
    for _ in range(args.num_runs):
        res = single_run(
                args,
                logger,
                env = env,
                eval_env = eval_env,
                num_episodes = args.num_episodes,
                lr_actor=args.lr_actor,
                lr_critic=args.lr_critic,
                pol_ent=args.pol_ent)
        res_PGT.append(res)

    # Save results
    returns_PGT = np.array([i[0] for i in res_PGT])
    samples_PGT = np.array([i[1] for i in res_PGT])
    np.save(logger.save_folder + '/', returns_PGT)
    logger.save_2(returns_PGT)

if __name__ == '__main__':
    main()

