# File: models/response_generator.py
import re
from typing import Dict

class ResponseGenerator:
    def __init__(self, article_db):
        self.article_db = article_db
        
    def generate_answer(self, user_question: str, article: Dict) -> Dict:
        """Генерация ответа на основе статьи"""
        
        # Извлекаем ключевую информацию из статьи
        key_points = self._extract_key_points(article['content'])
        
        # Форматируем ответ
        answer_text = self._format_answer(user_question, key_points, article)
        
        return {
            "answer": answer_text,
            "recommended_article": {
                "title": article['title'],
                "url": article['url'],
                "confidence": self._calculate_confidence(user_question, article)
            },
            "suggested_actions": self._get_suggested_actions()
        }
    
    def _extract_key_points(self, content: str, max_points: int = 3) -> list:
        """Извлечение ключевых пунктов из содержания статьи"""
        # Простая эвристика для извлечения ключевых пунктов
        sentences = content.split('.')
        key_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                any(keyword in sentence.lower() for keyword in ['важно', 'необходимо', 'следует', 'рекомендуется'])):
                key_sentences.append(sentence)
            
            if len(key_sentences) >= max_points:
                break
        
        # Если не нашли ключевые предложения, берем первые
        if not key_sentences and sentences:
            key_sentences = sentences[:max_points]
        
        return [s.strip() for s in key_sentences if s.strip()]
    
    def _format_answer(self, question: str, key_points: list, article: Dict) -> str:
        """Форматирование итогового ответа"""
        answer = f"Вот ответ на ваш вопрос '{question}':\n\n"
        
        if key_points:
            answer += "Основные шаги:\n"
            for i, point in enumerate(key_points, 1):
                answer += f"{i}. {point}.\n"
        else:
            answer += "Вот что я нашел по вашему вопросу:\n"
            answer += f"Статья '{article['title']}' содержит релевантную информацию.\n"
        
        answer += f"\nПодробнее можете узнать здесь: {article['url']}"
        
        return answer
    
    def _calculate_confidence(self, question: str, article: Dict) -> float:
        """Вычисление уверенности в рекомендации"""
        # Простая эвристика на основе ключевых слов
        question_lower = question.lower()
        title_lower = article['title'].lower()
        
        # Проверяем совпадение ключевых слов
        question_words = set(re.findall(r'\w+', question_lower))
        title_words = set(re.findall(r'\w+', title_lower))
        
        common_words = question_words.intersection(title_words)
        
        if not common_words:
            return 0.3
        elif len(common_words) >= 3:
            return 0.9
        else:
            return 0.6
    
    def _get_suggested_actions(self) -> list:
        """Предлагаемые действия для пользователя"""
        return [
            "Задать уточняющий вопрос",
            "Получить дополнительную статью по теме", 
            "Оценить полезность ответа"
        ]