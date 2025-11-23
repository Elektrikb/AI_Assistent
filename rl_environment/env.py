# File: rl_environment/env.py
import numpy as np
from typing import Tuple, Dict, Any, List
import random
import logging

logger = logging.getLogger(__name__)

class RecommendationEnv:
    def __init__(self, article_db, state_encoder, config):
        self.article_db = article_db
        self.state_encoder = state_encoder
        self.config = config
        
        self.current_user_query = None
        self.conversation_history = []
        self.available_actions = list(range(len(article_db.get_all_articles())))
        
        # Обновляем размерность действий
        self.action_dim = len(self.available_actions)
        
        logger.info(f"Environment initialized with {self.action_dim} actions")
    
    def reset(self, user_query: str) -> np.ndarray:
        """Сброс среды для нового диалога"""
        self.current_user_query = user_query
        self.conversation_history = []
        
        state = self._get_state()
        logger.debug(f"Environment reset. State shape: {state.shape}")
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Выполнение действия (рекомендация статьи)"""
        if action not in self.available_actions:
            logger.warning(f"Invalid action: {action}. Using random action.")
            action = random.choice(self.available_actions)
        
        recommended_article = self.article_db.get_article(action)
        
        if not recommended_article:
            logger.error(f"Article with id {action} not found")
            # Возвращаем состояние с нулевым reward
            next_state = self._get_state()
            return next_state, self.config.reward_failure, True, {"error": "Article not found"}
        
        # Вычисляем reward на основе релевантности
        reward = self._calculate_reward(recommended_article)
        
        # Сохраняем взаимодействие в историю
        self.conversation_history.append({
            'user_query': self.current_user_query,
            'recommended_article_id': action,
            'recommended_article_title': recommended_article['title'],
            'reward': reward
        })
        
        # Получаем следующее состояние
        next_state = self._get_state()
        
        # Определяем завершение эпизода
        done = self._is_episode_done(reward)
        
        info = {
            'article_title': recommended_article['title'],
            'article_url': recommended_article['url'],
            'reward': reward
        }
        
        logger.debug(f"Step: action={action}, reward={reward:.3f}, done={done}")
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Получить текущее состояние"""
        return self.state_encoder.encode_state(
            self.current_user_query, 
            self.conversation_history
        )
    
    def _calculate_reward(self, article: Dict) -> float:
        """Вычисление вознаграждения за рекомендацию"""
        try:
            query_embedding = self.state_encoder.text_model.encode([self.current_user_query])[0]
            article_embedding = self.article_db.get_article_embedding(article['id'])
            
            if article_embedding is None:
                return self.config.reward_failure
            
            # Нормализуем векторы
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            article_embedding = article_embedding / np.linalg.norm(article_embedding)
            
            similarity = np.dot(query_embedding, article_embedding)
            
            # Преобразуем схожесть в reward
            if similarity > 0.6:
                return self.config.reward_success  # Отличная рекомендация
            elif similarity > 0.3:
                return self.config.reward_partial  # Удовлетворительная рекомендация
            else:
                return self.config.reward_failure  # Плохая рекомендация
                
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return self.config.reward_failure
    
    def _is_episode_done(self, reward: float) -> bool:
        """Определить завершение эпизода"""
        high_reward = reward >= self.config.reward_success * 0.8
        max_length_reached = len(self.conversation_history) >= self.config.max_conversation_length
        
        return high_reward or max_length_reached
    
    def get_action_space_size(self) -> int:
        """Получить размер пространства действий"""
        return self.action_dim