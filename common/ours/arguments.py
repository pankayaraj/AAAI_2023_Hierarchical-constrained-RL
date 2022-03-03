import argparse


def get_args():
    """
    Utility for getting the arguments from the user for running the experiment

    :return: parsed arguments
    """

    # Env
    parser = argparse.ArgumentParser(description='collect arguments')

<<<<<<< Updated upstream
=======
<<<<<<< HEAD
<<<<<<< HEAD
    parser.add_argument('--save_dir', type=str, default="results/grid/safe_global_hrl_sarsa_key/")
    parser.add_argument('--exp_no', type=str, default="5")
    parser.add_argument("--goal_space", type=list, default=[212, 86, 160, 163, 282, 135, 200])  #this is for dicrete goal setting in an HRL setup
=======
=======
>>>>>>> 084226d0c16e822dff78e2ab976e836a70a26810
>>>>>>> Stashed changes
    parser.add_argument('--save_dir', type=str, default="results/grid/safe_cost_alloc_lower_hrl_sarsa_key/")
    parser.add_argument('--exp_no', type=str, default="3")
    parser.add_argument("--goal_space", type=list, default=[212, 86, 160, 163, 282, 135, 200])  #this is for discrete goal setting in an HRL setup
    parser.add_argument("--cost_space", type=list, default=[0, 1, 2, 3])  # this is for discrete cost space
    parser.add_argument("--cost_mapping", type=list, default=[0.0, 0.3, 0.6, 0.9])  # this is for discrete cost space

<<<<<<< Updated upstream
=======
<<<<<<< HEAD
>>>>>>> 084226d0c16e822dff78e2ab976e836a70a26810
=======
>>>>>>> 084226d0c16e822dff78e2ab976e836a70a26810
>>>>>>> Stashed changes
    #[212, 86, 160, 163, 282, 135, 200]
    parser.add_argument('--env-name', default='pg',
                        help="pg: point gather env\n"\
                             "cheetah: safe-cheetah env\n"\
                             "grid: grid world env\n"\
                            "pc: point circle env\n" \
                             "grid_key: grid_world env with key\n" \
                        )

    parser.add_argument('--agent', default='ppo',
                        help="the RL algo to use\n"\
                             "ppo: for ppo\n"\
                             "lyp-ppo: for Lyapnunov based ppo\n" \
                             "bvf-ppo: for Backward value function based ppo\n" \
                             "sarsa: for n-step sarsa\n" \
                             "lyp-sarsa: for Lyapnunov based sarsa\n"\
                             "bvf-sarsa: for Backward Value Function based sarsa\n" \
                             "hrl-sarsa: for HRL n-step sarsa\n" \
                        )
    parser.add_argument('--gamma', type=float, default=0.99, help="discount factor")
    parser.add_argument('--d0', type=float, default=90.0, help="the threshold for safety")

    # Actor Critic arguments goes here
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                            help="learning rate")
    parser.add_argument('--target-update-steps', type=int, default=int(1e4),
                        help="number of steps after to train the agent")
    parser.add_argument('--beta', type=float, default=0.001, help='entropy regularization')
    parser.add_argument('--critic-lr', type=float, default=1e-3, help="critic learning rate")
    parser.add_argument('--updates-per-step', type=int, default=1, help='model updates per simulator step (default: 1)')
    parser.add_argument('--tau', type=float, default=0.001, help='soft update rule for target netwrok(default: 0.001)')

    # PPO arguments go here
    parser.add_argument('--num-envs', type=int, default=10, help='the num of envs to gather data in parallel')
    parser.add_argument('--ppo-updates', type=int, default=1, help='num of ppo updates to do')
    parser.add_argument('--gae', type=float, default=0.95, help='GAE coefficient')
    parser.add_argument('--clip', type=float, default=0.2, help='clipping param for PPO')
    parser.add_argument('--traj_len', type=int, default=10,
                        help="for non HRL algos")
    parser.add_argument('--traj_len_u', type=int, default= 1, help="upper level's maximum length of the trajectory for an update")
    parser.add_argument('--traj_len_l', type=int, default=10,
                        help="lower level's maximum length of the trajectory for an update")

    parser.add_argument('--early-stop', action='store_true',
                        help="early stop pi training based on target KL ")

    # Optmization arguments
    parser.add_argument('--lr', type=float, default=1e-2,
                            help="learning rate")
    parser.add_argument('--adam-eps', type=float, default=0.95, help="momenturm for RMSProp")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='size of minibatch for ppo/ ddpg update')

    # Safety params
    parser.add_argument('--cost-reverse-lr', type=float, default=5e-4,
                            help="reverse learning rate for reviewer")
    parser.add_argument('--cost-q-lr', type=float, default=5e-4,
                            help="reverse learning rate for critic")
    parser.add_argument('--cost_allocator_lr', type=float, default=5e-4,
                        help="cost_allocator learning rate for critic")


    parser.add_argument('--cost-sg-coeff', type=float, default=0.0,
                            help="the coeeficient for the safe guard policy, minimizes the cost")
    parser.add_argument('--prob-alpha', type=float, default=0.6,
                        help="the kappa parameter for the target networks")
    parser.add_argument('--target', action='store_true',
                        help="use the target network based implementation")

    # Training arguments
    parser.add_argument('--num-steps', type=int, default=int(1e4),
                        help="number of steps to train the agent")
    parser.add_argument('--num-episodes', type=int, default=int(3e5),
                        help="number of episodes to train the agetn")

    parser.add_argument('--max_ep_len_u', type=int, default=int(5),
                        help="number of steps in an episode")
    parser.add_argument('--max_ep_len_l', type=int, default=int(80),
                        help="number of steps in an episode")

    # Evaluation arguments
    parser.add_argument('--eval-every', type=float, default=1000,
                        help="eval after these many steps")
    parser.add_argument('--eval-n', type=int, default=3,
                        help="average eval results over these many episodes")

    # Experiment specific
    parser.add_argument('--gpu', action='store_true', help="use the gpu and CUDA")
    parser.add_argument('--log-mode-steps', action='store_true',
                            help="changes the mode of logging w.r.r num of steps instead of episodes")
    parser.add_argument('--log-every', type=int, default=100,
                        help="logging schedule for training")
    parser.add_argument('--checkpoint-interval', type=int, default=1e5,
                        help="when to save the models")
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=str, default='/tmp/safe/models/')
    parser.add_argument('--log-dir', type=str, default="/tmp/safe/logs/")
    parser.add_argument('--reset-dir', action='store_true',
                        help="give this argument to delete the existing logs for the current set of parameters")

    args = parser.parse_args()

    return args


