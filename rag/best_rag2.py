import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BestRAG2:
    def __init__(self, questions_path: str, websites_path: str):
        self.questions_path = questions_path
        self.websites_path = websites_path
        self.NO_ANSWER_THRESHOLD = 0.08
        self.TARGET_MAX_LENGTH = 500

    def clean(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def prepare_data(self):
        print("Загрузка данных...")

        websites_df = pd.read_csv(self.websites_path)

        self.knowledge_base = []
        for _, row in websites_df.iterrows():
            text = self.clean(row['text'])
            title = self.clean(row['title'])
            if text and len(text) > 50:
                # Для поиска: заголовок с бОльшим весом + текст
                search_text = (title + ' ') * 3 + text.lower()
                self.knowledge_base.append({
                    'web_id': row['web_id'],
                    'url': row['url'],
                    'title': title,
                    'text': text,
                    'search_text': search_text.lower()
                })

        print(f"База знаний: {len(self.knowledge_base)} документов")

        questions_df = pd.read_csv(self.questions_path)
        self.questions = [
            {
                'q_id': row['q_id'],
                'query': self.clean(row['query']),
                'query_lower': self.clean(row['query']).lower()
            }
            for _, row in questions_df.iterrows()
        ]
        print(f"Вопросов: {len(self.questions)}")

        # TF-IDF по поисковому тексту (заголовок + текст)
        print("Строим TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        search_texts = [item['search_text'] for item in self.knowledge_base]
        self.tfidf_matrix = self.vectorizer.fit_transform(search_texts)
        print(f"Матрица: {self.tfidf_matrix.shape}")

    def retrieve(self, query_lower: str, top_k: int = 5) -> list:
        query_vec = self.vectorizer.transform([query_lower])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(sims)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            doc = self.knowledge_base[idx].copy()
            doc['similarity'] = float(sims[idx])
            results.append(doc)
        return results

    def find_best_passage(self, query: str, text: str) -> str:
        """Находим лучший абзац - скользящее окно по предложениям"""
        query_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())

        # Убираем слишком короткие/общие слова
        query_words = {w for w in query_words if len(w) > 2}

        sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', text) if len(s.strip()) > 15]

        if not sentences:
            return text[:self.TARGET_MAX_LENGTH]

        # Скорим каждое предложение
        scores = []
        for s in sentences:
            s_words = set(re.sub(r'[^\w\s]', '', s.lower()).split())
            overlap = len(query_words & s_words)
            scores.append(overlap)

        # Скользящее окно размером 5 предложений
        window = 5
        best_start = 0
        best_score = -1
        for i in range(len(sentences)):
            w_score = sum(scores[i:i + window])
            if w_score > best_score:
                best_score = w_score
                best_start = i

        # Берем окно вокруг лучшего места
        passage = ' '.join(sentences[best_start:best_start + window])

        # Обрезаем до нужной длины
        if len(passage) > self.TARGET_MAX_LENGTH:
            passage = passage[:self.TARGET_MAX_LENGTH]
            last_dot = max(passage.rfind('.'), passage.rfind('!'), passage.rfind('?'))
            if last_dot > self.TARGET_MAX_LENGTH // 2:
                passage = passage[:last_dot + 1]

        return passage.strip()

    def generate_answer(self, query: str, retrieved_docs: list) -> str:
        best_doc = retrieved_docs[0]
        best_sim = best_doc['similarity']

        # Нет ответа - если сходство низкое
        if best_sim < self.NO_ANSWER_THRESHOLD:
            return "Нет ответа."

        # Ищем лучший абзац
        passage = self.find_best_passage(query, best_doc['text'])

        # Если слишком короткий - берем из второго документа
        if len(passage) < 100 and len(retrieved_docs) > 1:
            passage2 = self.find_best_passage(query, retrieved_docs[1]['text'])
            if len(passage2) > len(passage):
                passage = passage2

        if len(passage.strip()) < 20:
            return "Нет ответа."

        return passage.strip()

    def process(self, output_path: str = "best_submission2.csv"):
        self.prepare_data()

        print(f"Генерация ответов для {len(self.questions)} вопросов...")
        results = []

        for i, q in enumerate(self.questions):
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(self.questions)}")

            try:
                retrieved = self.retrieve(q['query_lower'], top_k=5)
                answer = self.generate_answer(q['query'], retrieved)
            except Exception:
                answer = "Нет ответа."

            results.append({'q_id': q['q_id'], 'answer_new': answer})

        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, encoding='utf-8')

        lengths = df['answer_new'].str.len()
        no_answer = (df['answer_new'] == 'Нет ответа.').sum()

        print(f"\n=== СТАТИСТИКА ===")
        print(f"Нет ответа: {no_answer} ({no_answer/len(df)*100:.1f}%) | Цель: 32.7%")
        print(f"Средняя длина: {lengths.mean():.0f} | Цель: 387")
        print(f"Медиана: {lengths.median():.0f} | Цель: 262")
        print(f"\n✅ Файл: {output_path}")

        return df

if __name__ == "__main__":
    rag = BestRAG2("questions.csv", "websites.csv")
    rag.process("best_submission2.csv")
