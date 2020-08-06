import numpy as np

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

def objective(params, lmbda):
    """Objective for exact solution"""
    jtheta = policy_performance(params, utils.softmax, mdp, (1-gamma)*initial_distribution, args)
    reg = entropy_regularizer(params, utils.softmax, mdp, initial_distribution)
    return jtheta - lmbda*reg

def exact_solution(args, mdp, plot=True):
    """Compute the true solution with Value Iteration and Exact Policy Gradient"""
    print ("Computing Value Iteration and Exact Policy Gradient")
    print ("Environment", args.env)

    val_grad = value_and_grad(objective)

    lr = 0.1
    num_iterations = args.num_episodes
    P, R, gamma, initial_distribution = mdp
    lmbdas = [0, 0.1, 1.0]
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
            v_eval_items.append(policy_performance(l, utils.softmax, mdp, (1-gamma)*initial_distribution, args))

        # if not (os.path.exists("exact_result" + "/" + args.env  + "/" + str(items))):
        #   os.makedirs("exact_result" + "/" + args.env + "/" + str(items))
        # np.save("exact_result" + "/" + args.env + "/" + str(items)+"/result.npy", v_eval_items)
        v_eval_all.append(v_eval_items)

    vf, _ = value_iteration(P, R, gamma, num_iters=100)
    v_fin= ((1-gamma)*initial_distribution).T @ vf
    # v_fin = (initial_distribution).T @ vf
    v_fin_res = v_fin*np.ones_like(v_eval_all[0])
    # if not (os.path.exists("exact_result" + "/" + args.env + "/vi")):
    #   os.makedirs("exact_result" + "/" + args.env + "/vi")
    # np.save("exact_result" + "/" + args.env + "/vi/result.npy", v_fin_res)

    if plot:
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

