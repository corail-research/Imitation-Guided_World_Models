import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from agent.memory.DreamerMemory import DreamerMemory
from agent.models.DreamerModel import DreamerModel
from agent.optim.loss import model_loss, actor_loss, value_loss, actor_rollout, supervised_actor_rollout, supervised_actor_loss, hybrid_actor_rollout, hybrid_actor_loss
from agent.optim.utils import advantage
from environments import Env
from networks.dreamer.action import Actor
from networks.dreamer.critic import AugmentedCritic

from agent.optim.utils import compute_return


def orthogonal_init(tensor, gain=1):
    """
    Initialize a tensor with orthogonal values.
    :param tensor: Tensor to initialize
    :param gain: Scaling factor for the initialization
    :return: Tensor with orthogonal initialization
    """
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, _, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, scale=1.0, mode='ortho'):
    """
    Initialize the weights of a module.
    :param mod: Module whose weights to initialize
    :param scale: Scaling factor for the initialization
    :param mode: Initialization mode ('ortho' or 'xavier')
    :return: None
    """
    for p in mod.parameters():
        if mode == 'ortho':
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
        elif mode == 'xavier':
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)


class DreamerLearner:
    """
    Dreamer Learner that manages the model, actor, and critic.
    It initializes the model, actor, and critic, and provides methods for training.
    """
    def __init__(self, config):
        self.config = config
        if self.config.ABLATION_WM:
            self.actor = Actor(config.IN_DIM, config.ACTION_SIZE,
                            config.ACTION_HIDDEN, config.ACTION_LAYERS).to(
                config.DEVICE)
            self.critic = AugmentedCritic(config.IN_DIM, config.HIDDEN).to(config.DEVICE)
        else:
            self.model = DreamerModel(config).to(config.DEVICE).eval()
            self.actor = Actor(config.FEAT, config.ACTION_SIZE,
                            config.ACTION_HIDDEN, config.ACTION_LAYERS).to(
                config.DEVICE)
            self.critic = AugmentedCritic(config.FEAT, config.HIDDEN).to(config.DEVICE)
            initialize_weights(self.model, mode='xavier')
        initialize_weights(self.actor)
        initialize_weights(self.critic, mode='xavier')
        self.old_critic = deepcopy(self.critic)
        self.replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH,
                                           config.ACTION_SIZE, config.IN_DIM, 2,
                                           config.DEVICE, config.ENV_TYPE)
        self.entropy = config.ENTROPY
        self.step_count = -1
        self.cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0
        self.init_optimizers()
        self.optimizers_updated = False
        self.n_agents = 2
        Path(config.LOG_FOLDER).mkdir(parents=True, exist_ok=True)
        global wandb
        import wandb
        wandb.init(dir=config.LOG_FOLDER)

    def init_optimizers(self, factor=1.0):
        """
        Initialize the optimizers for the model, actor, and critic.
        :param factor: Scaling factor for the learning rates
        :return: None
        """
        if self.config.ABLATION_WM:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                    lr=self.config.ACTOR_LR*factor, 
                                                    weight_decay=0.00001)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                    lr=self.config.VALUE_LR*factor)
        else:
            self.model_optimizer = torch.optim.Adam(self.model.parameters(),
                                                    lr=self.config.MODEL_LR*factor)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                    lr=self.config.ACTOR_LR*factor, 
                                                    weight_decay=0.00001)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                    lr=self.config.VALUE_LR*factor)
        self.supervised_actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                        lr=self.config.ACTOR_LR*factor,
                                                        weight_decay=0.00001)

    def params(self):
        """
        Get the parameters of the model, actor, and critic.
        :return: Dictionary containing the state dictionaries of the model, actor, and critic
        """
        if self.config.ABLATION_WM:
            return {'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                    'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()}}
            
        return {'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()}}

    def save_params(self, saving_name):
        """
        Save the parameters of the model, actor, and critic to a file.
        :param saving_name: Name of the file to save the parameters
        :return: None
        """
        params = self.params()
        saving_name = saving_name+".pth"
        torch.save(params, saving_name)
        print("Learner saved")

    def load_params(self, path):
        """
        Load the parameters of the model, actor, and critic from a file.
        :param path: Path to the file containing the parameters
        :return: None
        """
        params = torch.load(path, map_location=self.config.DEVICE)
        if self.config.ABLATION_WM:
            self.actor.load_state_dict(params['actor'])
            self.critic.load_state_dict(params['critic'])
            print("Learner loaded from", path)
        else:
            self.model.load_state_dict(params['model'])
            self.actor.load_state_dict(params['actor'])
            self.critic.load_state_dict(params['critic'])
            print("Learner loaded from", path)

    def step(self, rollout, cur_step, max_step):
        """
        Process a step of the learner, updating the model and training the agent.
        :param rollout: Rollout data containing observations, actions, rewards, etc.
        :param cur_step: Current step in the training process
        :param max_step: Maximum number of steps in the training process
        :return: None
        """
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])
        self.total_samples += len(rollout['action'])
        self.replay_buffer.append(rollout['observation'],
                                  rollout['action'],
                                  rollout['reward'],
                                  rollout['done'],
                                  rollout['fake'],
                                  rollout['last'],
                                  rollout.get('avail_action'),
                                  rollout['expert_action'])
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if len(self.replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = 0
        sys.stdout.flush()

        if not self.config.ABLATION_WM:
            print("With world model")
            for i in range(self.config.MODEL_EPOCHS):
                samples = self.replay_buffer.sample(self.config.MODEL_BATCH_SIZE)
                self.train_model(samples,
                                    self.config.USE_SEQUENTIAL_IMITATION,
                                    self.config.USE_HYBRID_IMITATION)
        else:
            print("Without world model")

        if self.config.USE_SEQUENTIAL_IMITATION:
            if cur_step <= self.config.STEPS_SEQUENTIAL_SUPERVISED_PHASE: 
                print("Training in a supervised fashion")
                for i in range(self.config.EPOCHS):
                    samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
                    self.train_supervised_agent(samples)
            else:
                if not self.optimizers_updated:
                    self.init_optimizers(factor=1.0) 
                    self.optimizers_updated = True
                print("Training in a reinforcement fashion")
                for i in range(self.config.EPOCHS):
                    samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
                    self.train_agent(samples)
        elif self.config.USE_HYBRID_IMITATION:
            for i in range(self.config.EPOCHS):
                samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
                self.train_hybrid_agent(samples, cur_step, max_step)
        elif self.config.USE_DAGGER:
            samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
            self.train_supervised_agent(samples)
        else:
            for i in range(self.config.EPOCHS):
                samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
                self.train_agent(samples)

    def train_model(self, samples, use_sequential_imitation, use_hybrid_imitation):
        """
        Train the model using the provided samples.
        :param samples: Samples from the replay buffer
        :param use_sequential_imitation: Whether to use sequential imitation learning
        :param use_hybrid_imitation: Whether to use hybrid imitation learning
        :return: None
        """
        self.model.train()
        loss = model_loss(self.config, self.model, samples['observation'],
                          samples['action'], samples['av_action'],
                          samples['reward'], samples['done'], samples['fake'],
                          samples['last'], samples['expert_action'],
                          use_sequential_imitation, use_hybrid_imitation)
        self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.GRAD_CLIP)
        self.model.eval()

    def train_agent(self, samples):
        """
        Train the agent using the provided samples.
        :param samples: Samples from the replay buffer
        :return: None
        """
        if self.config.ABLATION_WM:
            obs = samples['observation']
            actions = samples['action']
            nb_agents = actions.shape[2]
            av_actions = samples['av_action']
            value = self.critic(obs)

            reward = samples['reward'][:-1]  
            reward = reward.mean(dim=2, keepdim=True)
            reward = reward.reshape(-1, 1, 1) 
            reward = reward.unsqueeze(0)

            value = value[:-1]
            value = value.reshape(-1, nb_agents, 1) 
            value = value.unsqueeze(0)

            discount_arr = torch.full_like(reward, 0.99)
            returns = compute_return(reward, value, discount_arr, bootstrap=value[-1], lmbda=self.config.DISCOUNT_LAMBDA,
                             gamma=self.config.GAMMA)  
            _, old_policy = self.actor(obs)
            
            features = obs[:-1].reshape(-1, nb_agents, 205).detach()
            returns = returns.squeeze(0).detach()
            actions = actions[:-1].reshape(-1, nb_agents, 3).detach()
            old_policy = old_policy[:-1].reshape(-1, nb_agents, 3).detach()

        else:
            actions, av_actions, old_policy, imag_feat, returns = actor_rollout(samples['observation'],
                                                                                samples['action'],
                                                                                samples['last'],
                                                                                self.model,
                                                                                self.actor,
                                                                                self.old_critic,
                                                                                self.config)
            features = imag_feat

        adv = returns.detach() - self.critic(features).detach()
        if self.config.ENV_TYPE == Env.STARCRAFT:
            adv = advantage(adv)
        wandb.log({'Agent/Returns': returns.mean()})
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            step = 2000
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]
                loss = actor_loss(features[idx], actions[idx],
                                  av_actions[idx] if av_actions is not None else None,
                                  old_policy[idx], adv[idx], self.actor, self.entropy)
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, 
                                     self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING
                val_loss = value_loss(self.critic, features[idx], returns[idx])
                self.apply_optimizer(self.critic_optimizer, self.critic, 
                                     val_loss, self.config.GRAD_CLIP_POLICY)
                if np.random.randint(20) == 9:
                    wandb.log({'Agent/val_loss': val_loss, 'Agent/actor_loss': loss})
                    wandb.log({'Agent/actor_optimizer_lr': self.actor_optimizer.param_groups[0]["lr"]})
                    wandb.log({'Agent/critic_optimizer_lr': self.critic_optimizer.param_groups[0]["lr"]})

                if self.config.ENV_TYPE == Env.FLATLAND and \
                    self.cur_update % self.config.TARGET_UPDATE == 0:
                    self.old_critic = deepcopy(self.critic)

    def train_supervised_agent(self, samples):
        """
        Train the agent using supervised learning with the provided samples.
        :param samples: Samples from the replay buffer
        :return: None
        """
        if self.config.ABLATION_WM:
            state = samples['observation']
            expert_action = samples['expert_action']
        else:
            state, expert_action = supervised_actor_rollout(samples['observation'],
                                                            samples['action'],
                                                            samples['last'], 
                                                            self.model,
                                                            samples['expert_action'])
        for epoch in range(self.config.SUPERVISED_EPOCHS):
            inds = np.random.permutation(expert_action.shape[0])
            step = 10
            for i in range(0, len(inds), step):
                idx = inds[i:i + step]
                loss = supervised_actor_loss(state[idx], expert_action[idx], self.actor)
                self.apply_optimizer(self.supervised_actor_optimizer, 
                                     self.actor, loss, self.config.GRAD_CLIP_POLICY)
                if np.random.randint(20) == 9:
                    wandb.log({'Agent/supervised_actor_loss': loss})
                    wandb.log({'Agent/supervised_actor_optimizer_lr': self.supervised_actor_optimizer.param_groups[0]["lr"]})

    def train_hybrid_agent(self, samples, cur_steps, max_steps):
        """
        Train the agent using hybrid imitation learning with the provided samples.
        :param samples: Samples from the replay buffer
        :param cur_steps: Current steps in the training process
        :param max_steps: Maximum steps in the training process
        :return: None
        """
        actions, av_actions, old_policy, imag_feat, returns, real_feat = hybrid_actor_rollout(samples['observation'],
                                                                            samples['action'],
                                                                            samples['last'],
                                                                            self.model,
                                                                            self.actor,
                                                                            self.old_critic,
                                                                            self.config)
        T, B_r, N, D   = real_feat.shape
        expert_actions = samples['expert_action'][:T]
        _, _, _, A   = expert_actions.shape

        real_feat_flat = real_feat.reshape(T*B_r, N, D)
        expert_actions_flat = expert_actions.reshape(T * B_r, N, A)

        adv = returns.detach() - self.critic(imag_feat).detach()
        if self.config.ENV_TYPE == Env.STARCRAFT:
            adv = advantage(adv)
        wandb.log({'Agent/Returns': returns.mean()})
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            step = 2000
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]
                loss = hybrid_actor_loss(imag_feat[idx], actions[idx], 
                                         av_actions[idx] if av_actions is not None else None,
                                         old_policy[idx], adv[idx], self.actor, self.entropy, 
                                         expert_actions_flat, real_feat_flat, cur_steps, max_steps)
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, 
                                     self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING
                val_loss = value_loss(self.critic, imag_feat[idx], returns[idx])
                if np.random.randint(20) == 9:
                    wandb.log({'Agent/val_loss': val_loss, 'Agent/actor_loss': loss})
                self.apply_optimizer(self.critic_optimizer, self.critic, 
                                     val_loss, self.config.GRAD_CLIP_POLICY)
                if self.config.ENV_TYPE == Env.FLATLAND and \
                   self.cur_update % self.config.TARGET_UPDATE == 0:
                    self.old_critic = deepcopy(self.critic)

    def apply_optimizer(self, opt, model, loss, grad_clip):
        """
        Apply the optimizer to the model based on the computed loss.
        :param opt: Optimizer to apply
        :param model: Model to optimize
        :param loss: Computed loss for the model
        :param grad_clip: Gradient clipping value
        :return: None
        """
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        
