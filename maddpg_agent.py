import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
from model import Actor, Critic


class MADDPGAgents:

    def __init__(self, state_size, action_size, n_agents, random_seed=7, device='cpu', buffer_size=1000000, batch_size=128, gamma=0.99, tau=0.001, lr_actor=0.0001, lr_critic=0.001, weight_decay=0):
        """
        Params
        ======
            state_size (int): dimension of the state space
            action_size (int): dimension of the action space
            n_agents (int): number of agents
            random_seed (int): random seed
            device (str): use cpu or gpu
            buffer_size (int): size of the replay buffer
            batch_size (int): size of each training batch
            gamma (float): discount factor for Q-value calculation
            tau (float): for soft update
            lr_action (float): learning rate for the actor networks
            lr_critic (float): learning rate for the critic networks
            weight_decay (float): for L2 regularization
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.seed = random.seed(random_seed)
        self.batch_size = batch_size
        self.device = device
        self.agents = [DDPGAgents(self.state_size, self.action_size, 1, random_seed=random_seed, device=device, gamma=gamma, tau=tau, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay)]*n_agents
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed, device)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, states, add_noise=True):
        actions = [a.act(np.expand_dims(s, axis=0), add_noise=add_noise) for a, s in zip(self.agents, states)]
        return actions
 
    def step(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            for agent in self.agents:
                experiences = self.memory.sample()
                agent.learn(experiences)


class DDPGAgents():
    
    def __init__(self, state_size, action_size, n_agents, random_seed=7, device='cpu', gamma=0.99, tau=0.001, lr_actor=0.0001, lr_critic=0.001, weight_decay=0):
        """
        Params
        ======
            state_size (int): dimension of the state space
            action_size (int): dimension of the action space
            n_agents (int): number of agents
            random_seed (int): random seed
            device (str): use cpu or gpu
            gamma (float): discount factor for Q-value calculation
            tau (float): for soft update
            lr_action (float): learning rate for the actor networks
            lr_critic (float): learning rate for the critic networks
            weight_decay (float): for L2 regularization
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.seed = random.seed(random_seed)
        self.gamma = gamma
        self.tau = tau
        self.device = device
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)
        # Noise process
        self.noise = OUNoise((n_agents, action_size), random_seed)

    def act(self, states, add_noise=True):
        states = torch.from_numpy(states).float().to(self.device)
        actions = np.zeros((self.n_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            for agent_num, state in enumerate(states):
                action = self.actor_local(state).cpu().data.numpy()
                actions[agent_num, :] = action
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        actions = np.clip(actions, -1, 1) 
        # if self.n_agents==1:
        #     actions = actions[0]   
        return actions

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

        self.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            

class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
    

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)