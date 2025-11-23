# File: database/article_db.py
import json
import numpy as np
from typing import List, Dict, Optional
import os
import logging
from .excel_loader import ExcelArticleLoader

logger = logging.getLogger(__name__)

class ArticleDatabase:
    def __init__(self, articles_path: str, excel_path: Optional[str] = None):
        self.articles_path = articles_path
        self.excel_path = excel_path
        self.articles = self._load_articles()
        
        # Инициализируем энкодер только если есть статьи
        if self.articles:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.article_embeddings = self._encode_articles()
        else:
            self.encoder = None
            self.article_embeddings = np.array([])
    
    def _load_articles(self) -> List[Dict]:
        """Загрузка статей из JSON или Excel"""
        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(self.articles_path), exist_ok=True)
        
        # Пробуем загрузить из JSON
        try:
            if os.path.exists(self.articles_path) and os.path.getsize(self.articles_path) > 0:
                with open(self.articles_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        articles = json.loads(content)
                        logger.info(f"Loaded {len(articles)} articles from JSON")
                        return articles
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Error loading from JSON: {e}")
        
        # Если есть Excel файл, загружаем из него
        if self.excel_path and os.path.exists(self.excel_path):
            logger.info(f"Loading articles from Excel: {self.excel_path}")
            excel_loader = ExcelArticleLoader(self.excel_path)
            articles = excel_loader.load_articles_from_excel()
            
            # Сохраняем в JSON для будущего использования
            if articles:
                self._save_articles(articles)
                return articles
        
        # Fallback: создаем sample articles
        return self._create_sample_articles()
    
    def _create_sample_articles(self) -> List[Dict]:
        """Создание примерных статей"""
        sample_articles = [
            {
                "id": 0,
                "title": "Как начать программировать на Python",
                "content": "Python - отличный язык для начала программирования. Установите Python с официального сайта, выберите среду разработки seperti PyCharm или VS Code. Изучите базовые конструкции: переменные, циклы, условия. Практикуйтесь на небольших проектах и изучайте стандартную библиотеку.",
                "url": "https://habr.com/ru/articles/490754/",
                "tags": ["python", "programming", "beginner", "установка"]
            },
            {
                "id": 1,
                "title": "Основы машинного обучения",
                "content": "Машинное обучение - это подраздел искусственного интеллекта. Изучите основные алгоритмы: линейная регрессия, логистическая регрессия, деревья решений. Важно понимать разницу между обучением с учителем и без учителя. Практикуйтесь на датасетах из Kaggle.",
                "url": "https://habr.com/ru/articles/31180/",
                "tags": ["machine learning", "ai", "algorithms", "обучение"]
            },
            {
                "id": 2, 
                "title": "Рекомендательные системы на практике",
                "content": "Рекомендательные системы помогают пользователям находить релевантный контент. Рассмотрим коллаборативную фильтрацию, контентные методы и гибридные подходы. Важные метрики: precision, recall, NDCG. Современные системы используют deep learning.",
                "url": "https://habr.com/ru/articles/448892/",
                "tags": ["recommendation systems", "collaborative filtering", "algorithms"]
            }
        ]
        
        # Сохраняем sample articles
        try:
            with open(self.articles_path, 'w', encoding='utf-8') as f:
                json.dump(sample_articles, f, ensure_ascii=False, indent=2)
            logger.info(f"Created sample articles at {self.articles_path}")
        except Exception as e:
            logger.error(f"Error saving sample articles: {e}")
        
        return sample_articles
    
    def _save_articles(self, articles: List[Dict]):
        """Сохранение статей в JSON"""
        try:
            with open(self.articles_path, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(articles)} articles to JSON")
        except Exception as e:
            logger.error(f"Error saving articles: {e}")
    
    def _encode_articles(self) -> np.ndarray:
        """Создание эмбеддингов для всех статей"""
        if not self.articles:
            return np.array([])
            
        try:
            texts = [f"{article['title']} {article['content'][:500]}" for article in self.articles]
            embeddings = self.encoder.encode(texts)
            logger.info(f"Encoded {len(embeddings)} article embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding articles: {e}")
            return np.array([])
    
    def get_article(self, article_id: int) -> Optional[Dict]:
        """Получить статью по ID"""
        if 0 <= article_id < len(self.articles):
            return self.articles[article_id]
        logger.warning(f"Article ID {article_id} not found")
        return None
    
    def get_article_embedding(self, article_id: int) -> Optional[np.ndarray]:
        """Получить эмбеддинг статьи"""
        if (0 <= article_id < len(self.article_embeddings) and 
            len(self.article_embeddings) > 0):
            return self.article_embeddings[article_id]
        return None
    
    def get_all_articles(self) -> List[Dict]:
        """Получить все статьи"""
        return self.articles
    
    def search_similar_articles(self, query: str, top_k: int = 5) -> List[Dict]:
        """Поиск похожих статей по запросу"""
        if not self.articles or len(self.article_embeddings) == 0:
            return []
            
        try:
            query_embedding = self.encoder.encode([query])
            similarities = np.dot(self.article_embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = [self.articles[i] for i in top_indices]
            logger.debug(f"Semantic search found {len(results)} articles for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return self.articles[:top_k]  # Fallback: возвращаем первые статьи