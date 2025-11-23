# File: database/session_manager.py
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import os

class SessionManager:
    def __init__(self, sessions_path: str):
        self.sessions_path = sessions_path
        self.sessions = self._load_sessions()
    
    def _load_sessions(self) -> Dict[str, Dict]:
        """Загрузка сессий из файла с обработкой ошибок"""
        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(self.sessions_path), exist_ok=True)
        
        try:
            # Проверяем существует ли файл и не пустой ли он
            if not os.path.exists(self.sessions_path) or os.path.getsize(self.sessions_path) == 0:
                return {}
            
            with open(self.sessions_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:  # Если файл пустой
                    return {}
                
                return json.loads(content)
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading sessions: {e}. Starting with empty sessions.")
            return {}
    
    def _save_sessions(self):
        """Сохранение сессий в файл"""
        try:
            with open(self.sessions_path, 'w', encoding='utf-8') as f:
                json.dump(self.sessions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving sessions: {e}")
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Создание новой сессии"""
        if user_id is None:
            user_id = str(uuid.uuid4())
        
        self.sessions[user_id] = {
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'conversation_history': [],
            'total_reward': 0.0,
            'interaction_count': 0
        }
        
        self._save_sessions()
        return user_id
    
    def add_interaction(self, user_id: str, user_query: str, 
                       recommended_article: Dict, reward: float):
        """Добавление взаимодействия в историю сессии"""
        if user_id not in self.sessions:
            self.create_session(user_id)
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'recommended_article': {
                'id': recommended_article['id'],
                'title': recommended_article['title'],
                'url': recommended_article['url']
            },
            'reward': reward
        }
        
        self.sessions[user_id]['conversation_history'].append(interaction)
        self.sessions[user_id]['total_reward'] += reward
        self.sessions[user_id]['interaction_count'] += 1
        self.sessions[user_id]['updated_at'] = datetime.now().isoformat()
        
        self._save_sessions()
    
    def get_session_history(self, user_id: str) -> List[Dict]:
        """Получение истории сессии"""
        if user_id in self.sessions:
            return self.sessions[user_id]['conversation_history']
        return []
    
    def get_session_stats(self, user_id: str) -> Optional[Dict]:
        """Получение статистики сессии"""
        if user_id in self.sessions:
            session = self.sessions[user_id]
            avg_reward = (session['total_reward'] / session['interaction_count'] 
                         if session['interaction_count'] > 0 else 0)
            
            return {
                'user_id': user_id,
                'created_at': session['created_at'],
                'interaction_count': session['interaction_count'],
                'total_reward': session['total_reward'],
                'avg_reward': avg_reward
            }
        return None