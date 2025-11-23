# File: models/state_encoder.py
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

class StateEncoder:
    def __init__(self, article_db):
        self.article_db = article_db
        self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # Фиксируем размерность - используем только эмбеддинг запроса
        self.state_dim = 384  # Размерность all-MiniLM-L6-v2
    
    def encode_state(self, user_query: str, conversation_history: List[Dict]) -> np.ndarray:
        """Кодирование состояния для RL-агента"""
        # Используем только эмбеддинг текущего запроса для простоты
        query_embedding = self.text_model.encode([user_query])[0]
        
        # Нормализуем эмбеддинг
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        return query_embedding
    
    def get_state_dimension(self) -> int:
        """Получить размерность вектора состояния"""
        return self.state_dim