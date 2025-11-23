# File: training/trainer.py
import numpy as np
import logging
from typing import List
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLTrainer:
    def __init__(self, env, agent, session_manager, config):
        self.env = env
        self.agent = agent
        self.session_manager = session_manager
        self.config = config
        
        # Тренировочные данные (симулированные запросы)
        self.training_queries = [
            "Как начать программировать на Python?",
            "Что такое машинное обучение?",
            "Объясните основы глубокого обучения",
            "Как работают рекомендательные системы?",
            "В чем разница между supervised и unsupervised learning?",
            "Как установить PyTorch?",
            "Что такое reinforcement learning?",
            "Какие есть библиотеки для data science в Python?",
            "Как создать нейронную сеть?",
            "Что такое коллаборативная фильтрация?"
        ]
    
    def train(self, episodes: int = 1000):
        """Основной цикл обучения"""
        logger.info(f"Starting training for {episodes} episodes")
        
        episode_rewards = []
        
        for episode in range(episodes):
            # Выбираем случайный тренировочный запрос
            user_query = random.choice(self.training_queries)
            
            # Сбрасываем среду
            state = self.env.reset(user_query)
            total_reward = 0
            episode_history = []
            
            for step in range(self.config.max_conversation_length):
                # Агент выбирает действие
                action = self.agent.select_action(state, training=True)
                
                # Выполняем действие в среде
                next_state, reward, done, info = self.env.step(action)
                
                # Сохраняем переход
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # Обучение агента
                self.agent.learn(batch_size=32)
                
                # Обновляем статистику
                state = next_state
                total_reward += reward
                
                episode_history.append({
                    'step': step,
                    'action': action,
                    'reward': reward,
                    'article_title': info['article_title']
                })
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            
            # Логирование прогресса
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"Episode {episode + 1}, Average Reward: {avg_reward:.3f}, Epsilon: {self.agent.epsilon:.3f}")
        
        logger.info("Training completed")
        return episode_rewards
    
    def evaluate(self, test_queries: List[str] = None) -> Dict:
        """Оценка обученного агента"""
        if test_queries is None:
            test_queries = self.training_queries[:5]  # Используем подмножество для оценки
        
        evaluation_results = {
            'total_reward': 0,
            'successful_recommendations': 0,
            'total_recommendations': 0,
            'query_results': []
        }
        
        for query in test_queries:
            state = self.env.reset(query)
            query_reward = 0
            recommendations = []
            
            for step in range(3):  # Максимум 3 шага на запрос
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                query_reward += reward
                recommendations.append({
                    'article': info['article_title'],
                    'reward': reward,
                    'step': step
                })
                
                state = next_state
                if done:
                    break
            
            evaluation_results['total_reward'] += query_reward
            evaluation_results['total_recommendations'] += len(recommendations)
            
            # Считаем успешные рекомендации (reward > 0.5)
            successful = sum(1 for rec in recommendations if rec['reward'] > 0.5)
            evaluation_results['successful_recommendations'] += successful
            
            evaluation_results['query_results'].append({
                'query': query,
                'total_reward': query_reward,
                'recommendations': recommendations,
                'success_rate': successful / len(recommendations) if recommendations else 0
            })
        
        evaluation_results['success_rate'] = (
            evaluation_results['successful_recommendations'] / 
            evaluation_results['total_recommendations'] 
            if evaluation_results['total_recommendations'] > 0 else 0
        )
        
        return evaluation_results