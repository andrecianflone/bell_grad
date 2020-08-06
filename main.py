import numpy as np
import gym
from collections import deque
import matplotlib.pyplot as plt
import pyprind
import pandas

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from models import GradientNetwork,SigmoidPolicy,Critic

from envs.gridworld import GridworldEnv
from envs.windy_gridworld import WindyGridworldEnv
from envs.twostateMDP import twostateMDP
from scipy.stats import entropy
import argparse
from utils import Logger
from utils import create_folder

# TODO: we shouldn't preload any environments, this slows down debugging
from environments import build_twostate_MDP, mdp_imani_counterexample,\
        mdp_fig2d, build_gridworld, build_windy_gridworld, build_FrozenLake, \
        build_FrozenLake8, build_SB_example35, build_CliffWalking

from functools import partial
import autograd.numpy as np
from autograd import value_and_grad
from autograd.scipy.misc import logsumexp


def one_hot_ify(state, num_states):
    res = torch.zeros(1, num_states)
    res[0,state] = 1
    return res

def solve_mdp(P, R, gamma, initial_distribution, policy):
    """ Policy Evaluation Solver

    We denote by 'A' the number of actions, 'S' for the number of
    states.

    Args:
      P (numpy.ndarray): Transition function as (A x S x S) tensor
      R (numpy.ndarray): Reward function as a (S x A) tensor
      gamma (float): Scalar discount factor
      policies (numpy.ndarray): tensor of shape (S x A)

    Returns:
      tuple (vf, qf) where the first element is vector of length S and the second element contains
      the Q functions as matrix of shape (S x A).
    """
    nstates = P.shape[-1]
    ppi = np.einsum('ast,sa->st', P, policy)
    rpi = np.einsum('sa,sa->s', R, policy)

    vf = np.linalg.solve(np.eye(nstates) - gamma*ppi, rpi)
    qf = R + gamma*np.einsum('ast,t->sa', P, vf)

    q_pi = np.einsum('sa,sa->s', qf, policy)
    vf_vector_rewards = np.linalg.solve(np.eye(nstates) - gamma*ppi, q_pi)

    return vf, qf, vf_vector_rewards

def discounted_stationary_distribution(P, policy, initial_distribution, discount):
    """Solve the discounted stationary distribution equations
    Args:
        transition (numpy.ndarray): Transition kernel as a (A x S x S) tensor
        policy (numpy.ndarray): Policy as a (S x A) matrix
        initial_distribution (numpy.ndarray): Initial distribution as a (S,) vector
        discount (float): Discount factor
    Returns:
        numpy.ndarray: The discounted stationary distribution as a (S,) vector
    """
    ppi = np.einsum('ast,sa->st', P, policy)
    A = np.eye(ppi.shape[0]) - discount*ppi
    b = (1 - discount)*initial_distribution
    return np.linalg.solve(A.T, b)

def policy_performance(params, policyfn, mdp, initial_distribution, args):
    """Expected discounted return from an initial state distribution

    Args:
        params (np.ndarray): Parameters of the policy
        policyfn (callable): Unary callable mapping parameters to a (S x A) nd.array of probs.
        mdp (tuple): P, R, gamma
        initial_distribution (np.ndarray): Weight vector (sums to 1)

    Returns:
        float: Scalar measuring the performance of the policy.
    """

    if args.pg_bellman:
        ## returns vf with vector valued rewards
        _, _, vf = solve_mdp(*mdp, policyfn(params))

    else:
        ## returns vf with scalar rewards
        vf, _, _ = solve_mdp(*mdp, policyfn(params))

    return np.dot(initial_distribution, vf)

def entropy_regularizer(params, policyfn, mdp, initial_distribution):
  """ Entropy of the discounted stationary distribution

     Args:
        params (np.ndarray): Parameters of the policy
        policyfn (callable): Unary callable mapping parameters to a (S x A) nd.array of probs.
        mdp (tuple): P, R, gamma
        initial_distribution (np.ndarray): Weight vector (sums to 1)

     Returns:
        float: Scalar measuring the entropy of the discounted stationary distribution induced by the policy parameters.
  """
  P, _, gamma, _ = mdp
  dpi = discounted_stationary_distribution(P, policyfn(params), initial_distribution, gamma)
  return dpi.T @ np.log(dpi)

def softmax(vals, temp=1.):
    """Batch softmax
    Args:
        vals (np.ndarray): S x A. Applied row-wise
        t (float, optional): Defaults to 1.. Temperature parameter
    Returns:
        np.ndarray: S x A
    """
    return np.exp(  (1./temp) * vals - logsumexp(  (1./temp) * vals, axis=1, keepdims=True) )


#### Compute the true solutuon with Value Iteration and Exact Policy Gradient

print ("Computing Value Iteration and Exact Policy Gradient")
print ("Environment", args.env)

P, R, gamma, initial_distribution = mdp

lmbdas = [0, 0.1, 1.0]

def objective(params, lmbda):
    jtheta = policy_performance(params, softmax, mdp, (1-gamma)*initial_distribution, args)
    reg = entropy_regularizer(params, softmax, mdp, initial_distribution)
    return jtheta - lmbda*reg

val_grad = value_and_grad(objective)

lr = 0.1
num_iterations = args.num_episodes

logits_all = []
for items in lmbdas:
    logits_items = []
    logits = np.zeros((P.shape[-1], P.shape[0]))
    for _ in range(num_iterations):
        v, g = val_grad(logits,items)
        logits_items.append(np.copy(logits))
        logits += lr*g
    logits_items.append(np.copy(logits))
    logits_all.append(logits_items)



v_eval_all = []
for item_id, items in enumerate(lmbdas):
    v_eval_items = []
    for l in logits_all[item_id]:
        v_eval_items.append(policy_performance(l, softmax, mdp, (1-gamma)*initial_distribution, args))

    # if not (os.path.exists("exact_result" + "/" + args.env  + "/" + str(items))):
    #   os.makedirs("exact_result" + "/" + args.env + "/" + str(items))
    # np.save("exact_result" + "/" + args.env + "/" + str(items)+"/result.npy", v_eval_items)
    v_eval_all.append(v_eval_items)


def value_iteration(P, R, gamma, num_iters=10):
    """Value iteration for the Bellman optimality equations
    Args:
        P (np.ndarray): Transition function as (A x S x S) tensor
        R (np.ndarray): Reward function as a (S x A) matrix
        gamma (float): Discount factor
        num_iters (int, optional): Defaults to 10. Number of iterations
    Returns:
        tuple: value function and state-action value function tuple
    """
    nstates, nactions = P.shape[-1], P.shape[0]
    qf = np.zeros((nstates, nactions))
    for _ in range(num_iters):
        qf = R + gamma*np.einsum('ast,t->sa', P, np.max(qf, axis=1))
    return np.max(qf, axis=1), qf

vf, _ = value_iteration(P, R, gamma, num_iters=100)

v_fin= ((1-gamma)*initial_distribution).T @ vf
# v_fin = (initial_distribution).T @ vf

v_fin_res = v_fin*np.ones_like(v_eval_all[0])
# if not (os.path.exists("exact_result" + "/" + args.env + "/vi")):
#   os.makedirs("exact_result" + "/" + args.env + "/vi")
# np.save("exact_result" + "/" + args.env + "/vi/result.npy", v_fin_res)


plt.clf()
for item_id in range(len(lmbdas)):
    plt.plot(v_eval_all[item_id], label = lmbdas[item_id])
plt.plot(v_fin_res, label='VI')
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.title('Policy Gradient with Vector Valued Rewards')
plt.legend()
plt.show()

plt.clf()
plt.scatter(lmbdas,np.array(v_eval_all)[:,-1])
v_true = v_fin*np.ones_like(lmbdas)
plt.plot(lmbdas,v_true,label='True value')
plt.xlabel('Lambdas')
plt.ylabel('Value')
plt.title('Policy Gradient with Vector Valued Rewards')
plt.legend()
plt.show()

def single_run(env, eval_env,
                num_episodes = 100,
                num_eval_episodes = 25,
                max_steps = 200,
                γ = 1.0,
                lr_actor = 0.01,
                lr_critic = 0.05):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    return_run = np.zeros(num_episodes)
    samples_run = np.zeros(num_episodes)
    actor = SigmoidPolicy(num_states, num_actions)
    actor_opt = optim.Adam(actor.parameters(), lr=lr_actor)
    critic = Critic(num_states, num_actions)
    critic_opt = optim.Adam(critic.parameters(), lr=lr_critic)


    actor_params_sizes = torch.tensor(np.cumsum([0] + [len(t.flatten()) for t in list(actor.parameters())]))

    gradient_network = GradientNetwork(num_states, actor_params_sizes[-1])
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

print("Running")

def main():
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
        logger = Logger(experiment_name = policy_name, environment_name = args.env, folder = args.folder)
        logger.save_args(args)
        print ('Saving to', logger.save_folder)

    if args.env == 'GridWorld':
        env = GridworldEnv()
        eval_env = GridworldEnv()
        mdp = build_gridworld()
        args.env = "GridWorld"

    elif args.env == 'WindyGridWorld':
        env = WindyGridworldEnv()
        eval_env = WindyGridworldEnv()
        mdp = build_windy_gridworld()
        args.env = "WindyGridWorld"

    elif args.env == 'CliffWalking':
        env = gym.make("CliffWalking-v0")
        eval_env = gym.make("CliffWalking-v0")
    elif args.env == 'FrozenLake':
        env = gym.make("FrozenLake-v0")
        eval_env = gym.make("FrozenLake-v0")

        mdp = build_FrozenLake()
        args.env = "FrozenLake"

    elif args.env == 'FrozenLake8':
        env = gym.make("FrozenLake8x8-v0")
        eval_env = gym.make("FrozenLake8x8-v0")
    elif args.env == 'Taxi':
        env = gym.make("Taxi-v2")
        eval_env = gym.make("Taxi-v2")
    elif args.env=='twostateMDP':
        env = gym.make('twostateMDP-v0')
        eval_env=gym.make('twostateMDP-v0')
        mdp = mdp_fig2d()
        args.env = mdp_fig2d


    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    env.reset()



    res_PGT = [single_run(env = env, eval_env = eval_env, args.num_episodes = args.num_episodes, lr_actor=args.lr_actor, lr_critic=args.lr_critic) for _ in range(args.num_runs)]
    returns_PGT = np.array([i[0] for i in res_PGT])
    samples_PGT = np.array([i[1] for i in res_PGT])

    np.save(logger.save_folder + '/', returns_PGT)
    logger.save_2(returns_PGT)

if __name__ == '__main__':
    main()

