# File: training/pretrain.py
import numpy as np
import logging
from typing import List, Dict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pretrainer:
    def __init__(self, env, agent, article_db):
        self.env = env
        self.agent = agent
        self.article_db = article_db
        
        # Тренировочные данные с правильными ответами
        self.training_data = [
            {
                "question": "Как начать программировать на Python?",
                "correct_article_id": 0,
                "keywords": ["python", "программировать", "начать", "установка"]
            },
            {
                "question": "Что такое машинное обучение?",
                "correct_article_id": 1,
                "keywords": ["машинное обучение", "ml", "алгоритмы", "искусственный интеллект"]
            },
            {
                "question": "Как работают рекомендательные системы?",
                "correct_article_id": 2,
                "keywords": ["рекомендательные системы", "коллаборативная фильтрация", "рекомендации"]
            },
            {
                "question": "Как установить PyTorch?",
                "correct_article_id": 3,
                "keywords": ["pytorch", "установить", "глубокое обучение", "нейронные сети"]
            },
            {
                "question": "Что такое reinforcement learning?",
                "correct_article_id": 4,
                "keywords": ["reinforcement learning", "rl", "обучение с подкреплением"]
            },
            {
                "question": "Какие базы данных использовать?",
                "correct_article_id": 5,
                "keywords": ["базы данных", "sql", "postgresql", "mysql"]
            },
            {
                "question": "Основы компьютерных сетей",
                "correct_article_id": 6,
                "keywords": ["компьютерные сети", "tcp/ip", "http", "dns"]
            },
            {
                "question": "Алгоритмы и структуры данных",
                "correct_article_id": 7,
                "keywords": ["алгоритмы", "структуры данных", "сортировка", "поиск"]
            },
            {
                "question": "Что такое DevOps?",
                "correct_article_id": 8,
                "keywords": ["devops", "docker", "kubernetes", "ci/cd"]
            },
            {
                "question": "Веб-разработка на Django",
                "correct_article_id": 9,
                "keywords": ["django", "веб-разработка", "python", "фреймворк"]
            }
        ]
    
    def pretrain_with_supervised(self, episodes: int = 500):
        """Предварительное обучение с учителем"""
        logger.info("Starting supervised pretraining...")
        
        for episode in range(episodes):
            # Выбираем случайный тренировочный пример
            training_example = np.random.choice(self.training_data)
            question = training_example["question"]
            correct_action = training_example["correct_article_id"]
            
            # Сбрасываем среду
            state = self.env.reset(question)
            
            # Агент выбирает действие
            action = self.agent.select_action(state, training=True)
            
            # Вычисляем reward на основе правильного ответа
            if action == correct_action:
                reward = 1.0  # Высокий reward за правильный ответ
            else:
                # Reward based on semantic similarity
                correct_article = self.article_db.get_article(correct_action)
                chosen_article = self.article_db.get_article(action)
                
                if correct_article and chosen_article:
                    # Вычисляем схожесть между выбранной и правильной статьей
                    correct_embedding = self.article_db.get_article_embedding(correct_action)
                    chosen_embedding = self.article_db.get_article_embedding(action)
                    
                    if correct_embedding is not None and chosen_embedding is not None:
                        similarity = np.dot(correct_embedding, chosen_embedding) / (
                            np.linalg.norm(correct_embedding) * np.linalg.norm(chosen_embedding)
                        )
                        reward = max(0.1, similarity)  # Минимальный reward 0.1
                    else:
                        reward = 0.1
                else:
                    reward = -0.5  # Штраф за несуществующую статью
            
            # Сохраняем переход
            next_state = self.env._get_state()  # Получаем следующее состояние
            done = True  # Завершаем эпизод после одной рекомендации
            
            self.agent.store_transition(state, action, reward, next_state, done)
            self.agent.learn(batch_size=16)
            
            if (episode + 1) % 100 == 0:
                logger.info(f"Pretraining episode {episode + 1}, reward: {reward:.3f}")
    
    def evaluate_pretraining(self) -> Dict:
        """Оценка качества после предварительного обучения"""
        logger.info("Evaluating pretraining performance...")
        
        correct_predictions = 0
        total_reward = 0
        
        for example in self.training_data:
            question = example["question"]
            correct_action = example["correct_article_id"]
            
            state = self.env.reset(question)
            action = self.agent.select_action(state, training=False)
            
            if action == correct_action:
                correct_predictions += 1
                reward = 1.0
            else:
                reward = 0.0
            
            total_reward += reward
        
        accuracy = correct_predictions / len(self.training_data)
        avg_reward = total_reward / len(self.training_data)
        
        logger.info(f"Pretraining accuracy: {accuracy:.3f}")
        logger.info(f"Average reward: {avg_reward:.3f}")
        
        return {
            "accuracy": accuracy,
            "avg_reward": avg_reward,
            "correct_predictions": correct_predictions,
            "total_examples": len(self.training_data)
        }