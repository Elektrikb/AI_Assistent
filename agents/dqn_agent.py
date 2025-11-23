# File: agents/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class SimpleDQN(nn.Module):
    """Упрощенная нейронная сеть для DQN"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(SimpleDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config):
        self.config = config
        self.action_dim = action_dim
        self.epsilon = config.epsilon_start
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Networks
        self.policy_net = SimpleDQN(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net = SimpleDQN(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=1000)
        
        # Training state
        self.steps_done = 0
        
        logger.info(f"DQN Agent initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Выбор действия с использованием epsilon-greedy стратегии"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if training and random.random() < self.epsilon:
            # Случайное действие (exploration)
            action = random.randint(0, self.action_dim - 1)
            logger.debug(f"Random action: {action}, epsilon: {self.epsilon:.3f}")
        else:
            # Действие от policy network (exploitation)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
                logger.debug(f"Network action: {action}, max_q: {q_values.max().item():.3f}")
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Сохранить переход в memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self, batch_size: int = 32):
        """Обучение на batch из memory"""
        if len(self.memory) < batch_size:
            return
        
        # Выборка batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Конвертация в тензоры
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        # Текущие Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        # Loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
        
        # Update target network
        if self.steps_done % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.steps_done += 1
        
        if self.steps_done % 50 == 0:
            logger.info(f"Training step {self.steps_done}, loss: {loss.item():.4f}, epsilon: {self.epsilon:.3f}")
    
    def save(self, filepath: str):
        """Сохранение модели"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath: str):
        """Загрузка модели"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']