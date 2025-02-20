from abc import ABC, abstractmethod
import re
import string
from typing import List, Optional

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

from pymystem3 import Mystem # pip install pymystem

from langdetect import detect # pip install langdetect
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)


class BaseProcessor(ABC):
    @abstractmethod
    def __call__(self, text: str) -> str:
        pass

class LowerCaseProcessor(BaseProcessor):
    def __call__(self, text: str) -> str:
        return text.lower()

class PunctuationRemoverProcessor(BaseProcessor):
    def __call__(self, text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

class WhitespaceNormalizerProcessor(BaseProcessor):
    def __call__(self, text: str) -> str:
        return " ".join(text.split())

class NumberRemoverProcessor(BaseProcessor):
    def __call__(self, text: str) -> str:
        return re.sub(r'\d+', '', text)

class UrlRemoverProcessor(BaseProcessor):
    def __call__(self, text: str) -> str:
        return re.sub(r'http\S+|www.\S+', '', text)

class EmailRemoverProcessor(BaseProcessor):
    def __call__(self, text: str) -> str:
        return re.sub(r'\S+@\S+', '', text)

class HTMLTagsRemoverProcessor(BaseProcessor):
    def __call__(self, text: str) -> str:
        return re.sub(r'<.*?>', '', text)

class EmoticonRemoverProcessor(BaseProcessor):
    def __call__(self, text: str) -> str:
        # Простой пример удаления эмотиконов
        return re.sub(r'[:;=]-[()DP]', '', text)

class TextNormalizerProcessor(BaseProcessor):
    def __call__(self, text: str) -> str:
        # Нормализация повторяющихся букв (например, "hellooooo" -> "hello")
        return re.sub(r'(.)\1+', r'\1', text)
    

# --- Класс базового препроцессинга
class BasicProcessor(BaseProcessor):
    def __init__(self):
        self.processors = [
            LowerCaseProcessor(),
            PunctuationRemoverProcessor(),
            NumberRemoverProcessor(),
            WhitespaceNormalizerProcessor()
        ]

    def __call__(self, text: str) -> str:
        for processor in self.processors:
            text = processor(text)
        return text
    
# --- Обработка стоп-слов
class StopWordsRemoverProcessor(BaseProcessor):
    def __init__(self, language: str = 'english', extra_stopwords: Optional[List[str]]=None):
        nltk.download('stopwords')
        self.stop_words = set(nltk.corpus.stopwords.words(language))

        if extra_stopwords is not None:
            self.stop_words = self.stop_words | set(extra_stopwords)

    def __call__(self, text: str) -> str:
        words = text.split()
        return " ".join([word for word in words if word.lower() not in self.stop_words])
    

# --- Лемматизация и стэмминг

class StemmerProcessor(BaseProcessor):
    def __init__(self):
        self.stemmer = PorterStemmer()

    def __call__(self, text: str) -> str:
        words = text.split()
        return " ".join([self.stemmer.stem(word) for word in words])

class LemmatisationProcessor(BaseProcessor):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, text: str) -> str:
        words = text.split()
        return " ".join([self.lemmatizer.lemmatize(word) for word in words])
    

# --- Обработка английского языка ---

    
class EnglishStemmer(BaseProcessor):
    def __init__(self, stemmer_type='porter'):
        self.stemmer = PorterStemmer() if stemmer_type == 'porter' else LancasterStemmer()

    def __call__(self, text: str) -> str:
        return ' '.join([self.stemmer.stem(word) for word in text.split()])
    

# --- Обработка русского языка ---

class RussianLemmatizer(BaseProcessor):
    def __init__(self):
        self.mystem = Mystem()

    def __call__(self, text: str) -> str:
        lemmas = self.mystem.lemmatize(text)
        return ''.join(lemmas).strip()
    

class RussianNatashaProcessor(BaseProcessor):
    def init(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)

    def call(self, text: str) -> str:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        return ' '.join([token.lemma for token in doc.tokens])
    

class TextProcessor:

    def __init__(self, processors: Optional[List]=None):
        self.processors = processors

    def __call__(self, text: str) -> str:
        """
            Two Behaviours:

            A: Use processors from self.processors
            B: If no self.processors provided, then use default processing route
        """

        if self.processors is not None:
            processors = self.processors
        else:
            language = detect(text)
            if (type(language) == str) and (language == 'ru'):
                processors = [
                    BasicProcessor(),
                    StopWordsRemoverProcessor(language='russian'),
                    RussianLemmatizer()
                ]
            else:
                processors = [
                    BasicProcessor(),
                    StopWordsRemoverProcessor(language='english'),
                    EnglishStemmer()
                ]

        for processor in processors:
            text = processor(text)
        
        return text
