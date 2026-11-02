import os
import re
from typing import List

import nltk

# NLTK download check for Assignment 9's Docker container
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DocumentProcessing:
    def __init__(self):
        pass

    def run(self, document_path: str, strategy: str, **kwargs) -> List[str]:
        """
        Runs the specified chunking strategy on a document. (For Assignment 9)
        """
        if strategy == 'sentence':
            return self.sentence_chunking(document_path, **kwargs)
        elif strategy == 'fixed-length':
            return self.fixed_length_chunking(document_path, **kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def __read_text_file(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            return ""
        except Exception as e:
            print(f"An error occurred reading file {file_path}: {e}")
            return ""

    def trim_white_space(self, text: str) -> str:
        return ' '.join(text.split())

    def sentence_chunking(self, document_filename: str, num_sentences: int, overlap_size: int = 0) -> list:
        # This implementation remains robust
        text = self.__read_text_file(document_filename)
        if not text:
            return []
        text = self.trim_white_space(text)
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []

        chunks = []
        i = 0
        step = num_sentences - overlap_size
        if step <= 0:
            step = num_sentences

        while i < len(sentences):
            chunk = ' '.join(sentences[i:i + num_sentences])
            chunks.append(chunk)
            i += step
        return chunks

    def fixed_length_chunking(self, document_filename: str, chunk_size: int, overlap_size: int = 0) -> List[str]:
        # FIX 1: Add input validation to raise ValueError when overlap is invalid.
        if overlap_size >= chunk_size > 0:
            raise ValueError("overlap_size must be less than chunk_size.")
        
        text = self.__read_text_file(document_filename)
        if not text:
            return []
        text = self.trim_white_space(text)

        chunks = []
        start = 0
        step = chunk_size - overlap_size
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            
            # FIX 2: Check if the last full chunk has reached the end.
            if end >= len(text):
                break
            
            start += step
        return chunks