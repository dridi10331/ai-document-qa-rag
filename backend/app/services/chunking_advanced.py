"""
Advanced chunking strategies: semantic, recursive, heading-aware, table-aware.
"""

import re
from typing import Any

from app.services.embeddings import EmbeddingService


class SemanticChunker:
    """Chunk based on semantic similarity between sentences."""
    
    def __init__(self, embedding_service: EmbeddingService, similarity_threshold: float = 0.7):
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
    
    def chunk(self, text: str, max_chunk_size: int = 500) -> list[str]:
        """
        Chunk text based on semantic boundaries.
        
        Groups sentences with high semantic similarity together.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        # Get embeddings for all sentences
        embeddings = [self.embedding_service.embed(s) for s in sentences]
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(current_embedding, embeddings[i])
            
            # If similarity drops below threshold or chunk too large, start new chunk
            if similarity < self.similarity_threshold or len(" ".join(current_chunk)) > max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]
            else:
                current_chunk.append(sentences[i])
                # Update running average embedding
                current_embedding = self._average_embeddings([current_embedding, embeddings[i]])
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0
    
    def _average_embeddings(self, embeddings: list[list[float]]) -> list[float]:
        """Average multiple embeddings."""
        dim = len(embeddings[0])
        return [sum(emb[i] for emb in embeddings) / len(embeddings) for i in range(dim)]


class RecursiveChunker:
    """Recursively chunk text using multiple separators."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]
    
    def chunk(self, text: str) -> list[str]:
        """
        Recursively chunk text using hierarchical separators.
        
        Tries to split on paragraphs first, then sentences, then words.
        """
        return self._recursive_split(text, self.separators)
    
    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators."""
        if not separators:
            # Base case: no more separators, return as-is
            return [text] if text else []
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if not separator:
            # Empty separator means split by character
            return self._split_by_size(text)
        
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                current_chunk += split + separator
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If split itself is too large, recursively split it
                if len(split) > self.chunk_size:
                    chunks.extend(self._recursive_split(split, remaining_separators))
                    current_chunk = ""
                else:
                    current_chunk = split + separator
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_size(self, text: str) -> list[str]:
        """Split text into fixed-size chunks."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks


class HeadingAwareChunker:
    """Chunk based on document structure (headings, sections)."""
    
    def __init__(self, max_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str) -> list[str]:
        """
        Chunk text preserving heading structure.
        
        Keeps headings with their content, doesn't split across sections.
        """
        # Detect headings (Markdown-style)
        lines = text.split("\n")
        
        chunks = []
        current_section = []
        current_heading = None
        
        for line in lines:
            # Check if line is a heading
            if re.match(r'^#{1,6}\s+', line):
                # Save previous section
                if current_section:
                    section_text = "\n".join(current_section)
                    if len(section_text) > self.max_chunk_size:
                        # Section too large, split it
                        chunks.extend(self._split_large_section(section_text, current_heading))
                    else:
                        chunks.append(section_text)
                
                # Start new section
                current_heading = line
                current_section = [line]
            else:
                current_section.append(line)
        
        # Save last section
        if current_section:
            section_text = "\n".join(current_section)
            if len(section_text) > self.max_chunk_size:
                chunks.extend(self._split_large_section(section_text, current_heading))
            else:
                chunks.append(section_text)
        
        return chunks
    
    def _split_large_section(self, text: str, heading: str | None) -> list[str]:
        """Split large section while preserving heading."""
        # Simple split by size, prepend heading to each chunk
        chunks = []
        words = text.split()
        current_chunk = [heading] if heading else []
        
        for word in words:
            if len(" ".join(current_chunk)) + len(word) > self.max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [heading] if heading else []
            current_chunk.append(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


class TableAwareChunker:
    """Chunk text while preserving table structure."""
    
    def __init__(self, max_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str) -> list[str]:
        """
        Chunk text preserving table boundaries.
        
        Detects tables (Markdown or simple text tables) and keeps them intact.
        """
        # Detect table boundaries (simple heuristic: lines with | or multiple tabs)
        lines = text.split("\n")
        
        chunks = []
        current_chunk = []
        in_table = False
        table_lines = []
        
        for line in lines:
            is_table_line = "|" in line or "\t\t" in line
            
            if is_table_line:
                if not in_table:
                    # Save previous chunk
                    if current_chunk:
                        chunks.append("\n".join(current_chunk))
                        current_chunk = []
                    in_table = True
                table_lines.append(line)
            else:
                if in_table:
                    # End of table, save it as single chunk
                    chunks.append("\n".join(table_lines))
                    table_lines = []
                    in_table = False
                
                current_chunk.append(line)
                
                # Check if chunk too large
                if len("\n".join(current_chunk)) > self.max_chunk_size:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
        
        # Save remaining
        if table_lines:
            chunks.append("\n".join(table_lines))
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        return chunks
