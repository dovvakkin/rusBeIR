from abc import ABC, abstractmethod
import re
import string
from typing import List, Optional, Union

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

from langdetect import detect # pip install langdetect

from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
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
        return re.sub(r'[:;=]-[()DP]', '', text)
    

# --- Класс базового препроцессинга
class BasicProcessor(BaseProcessor):
    def __init__(self):
        self.processors = [
            LowerCaseProcessor(),
            PunctuationRemoverProcessor(),
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
class LemmatisationProcessor(BaseProcessor):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, text: str) -> str:
        words = text.split()
        return " ".join([self.lemmatizer.lemmatize(word) for word in words])
    

class StemmerProcessor(BaseProcessor):
    def __init__(self, stemmer_type='porter'):
        self.stemmer = PorterStemmer() if stemmer_type == 'porter' else LancasterStemmer()

    def __call__(self, text: str) -> str:
        return ' '.join([self.stemmer.stem(word) for word in text.split()])
    

# --- Обработка русского языка ---

class NatashaBaseProcessor(BaseProcessor):
    """
    Базовый процессор Natasha, который инициализирует все необходимые компоненты
    """
    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)

    def prepare_doc(self, text: str) -> Doc:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.parse_syntax(self.syntax_parser)
        doc.tag_ner(self.ner_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        return doc

class NatashaLemmatizer(NatashaBaseProcessor):
    """
    Лемматизация текста
    """
    def __call__(self, text: str) -> str:
        doc = self.prepare_doc(text)
        return ' '.join([token.lemma for token in doc.tokens])

class NatashaNormalizer(NatashaBaseProcessor):
    """
    Нормализация текста (приведение к нормальной форме)
    """
    def __call__(self, text: str) -> str:
        doc = self.prepare_doc(text)
        return ' '.join([token.normalized for token in doc.tokens if token.normalized])

class NatashaStopWordsRemover(NatashaBaseProcessor):
    """
    Удаление стоп-слов на основе частей речи
    """
    def __init__(self):
        super().__init__()
        self.stop_pos = {'PREP', 'CONJ', 'PRCL', 'INTJ'}  # предлоги, союзы, частицы, междометия

    def __call__(self, text: str) -> str:
        doc = self.prepare_doc(text)
        return ' '.join([token.text for token in doc.tokens 
                        if token.pos not in self.stop_pos])

class NatashaNamedEntityRemover(NatashaBaseProcessor):
    """
    Удаление именованных сущностей (имена, организации, локации и т.д.)
    """
    def __call__(self, text: str) -> str:
        doc = self.prepare_doc(text)
        spans_to_remove = set()
        for span in doc.spans:
            spans_to_remove.update(range(span.start, span.stop))
        
        return ' '.join([token.text for i, token in enumerate(doc.tokens)
                        if i not in spans_to_remove])

class NatashaPunctuationRemover(NatashaBaseProcessor):
    """
    Удаление знаков пунктуации
    """
    def __call__(self, text: str) -> str:
        doc = self.prepare_doc(text)
        return ' '.join([token.text for token in doc.tokens 
                        if token.pos != 'PUNCT'])

class NatashaNounExtractor(NatashaBaseProcessor):
    """
    Извлечение только существительных
    """
    def __call__(self, text: str) -> str:
        doc = self.prepare_doc(text)
        return ' '.join([token.text for token in doc.tokens 
                        if token.pos == 'NOUN'])

class NatashaVerbExtractor(NatashaBaseProcessor):
    """
    Извлечение только глаголов
    """
    def __call__(self, text: str) -> str:
        doc = self.prepare_doc(text)
        return ' '.join([token.text for token in doc.tokens 
                        if token.pos == 'VERB'])
    

class TextProcessor(BaseProcessor):

    def __init__(self, processors: Optional[List]=None):
        self.processors = processors

        if self.processors is None:
            self._ru_processors = [
                NatashaPunctuationRemover(),
                NatashaStopWordsRemover(),
                NatashaLemmatizer(),
                LowerCaseProcessor()
            ]

            self._en_processors = [
                BasicProcessor(),
                StopWordsRemoverProcessor(language='english'),
                StemmerProcessor()
            ]


    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
            Two Behaviours:

            A: Use processors from self.processors
            B: If no self.processors provided, then use default processing route
        """

        if self.processors is not None:
            processors = self.processors
        else:
            try:
                language = detect(text)
            except:
                language = 'en'
            if (type(language) == str) and (language == 'ru'):
                processors = self._ru_processors
            else:
                processors = self._en_processors

        for processor in processors:
            if type(text) == str:
                text = processor(text)
            else:
                text = [processor(subtext) for subtext in text]
        
        return text
