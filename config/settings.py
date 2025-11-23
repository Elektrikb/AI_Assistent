# File: config/settings.py
import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    state_dim: int = 384
    hidden_dim: int = 256
    action_dim: int = 10
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 32

@dataclass
class EnvironmentConfig:
    max_conversation_length: int = 5
    reward_success: float = 1.0
    reward_failure: float = -0.1
    reward_partial: float = 0.3

@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

class Config:
    model = ModelConfig()
    environment = EnvironmentConfig()
    api = APIConfig()
    
    # Пути к данным
    ARTICLES_PATH = "data/articles.json"
    EXCEL_PATH = "data/articles.xlsx"  # Новый путь к Excel файлу
    SESSIONS_PATH = "data/user_sessions.json"