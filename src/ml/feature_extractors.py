"""
Symbolic Feature Extraction for Risk Assessment
Implements "white box" linguistic features based on SE research.
"""
import re
import textstat
import spacy
from typing import Dict, List
import numpy as np


class SymbolicFeatureExtractor:
    """
    Extracts symbolic linguistic features from user story text.
    
    Features:
    - Readability metrics (Flesch, Gunning Fog, Lexical Density)
    - Ambiguity indicators (Modal verbs, Vague quantifiers, Passive voice)
    - Risk lexicons (SATD, Security, Complexity)
    
    Output: 15-dimensional feature vector
    """
    
    # Risk lexicons based on SE research
    SATD_KEYWORDS = [
        "hack", "fixme", "todo", "workaround", "temporary",
        "ugly", "hardcoded", "quick fix", "spaghetti"
    ]
    
    SECURITY_KEYWORDS = [
        "auth", "token", "jwt", "encrypt", "pii", "gdpr",
        "role", "permission", "injection", "xss", "secret",
        "oauth", "credential", "certificate"
    ]
    
    COMPLEXITY_KEYWORDS = [
        "legacy", "mainframe", "wrapper", "migration", "api",
        "synchronization", "handshake", "middleware", "integration",
        "refactor", "database"
    ]
    
    WEAK_MODALS = ["might", "could", "should", "may", "ought"]
    
    VAGUE_QUANTIFIERS = [
        "fast", "easy", "robust", "user-friendly", "seamless",
        "efficient", "many", "few", "several", "tbd", "appropriate",
        "suitable", "good", "better", "nice"
    ]
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize feature extractor.
        
        Args:
            spacy_model: Name of spaCy model to load
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"⚠ spaCy model '{spacy_model}' not found.")
            print("  Run: python -m spacy download en_core_web_sm")
            raise
        
        # Compile regex patterns for efficiency
        self._weak_modal_pattern = re.compile(
            r'\b(' + '|'.join(self.WEAK_MODALS) + r')\b',
            re.IGNORECASE
        )
        self._vague_pattern = re.compile(
            r'\b(' + '|'.join(self.VAGUE_QUANTIFIERS) + r')\b',
            re.IGNORECASE
        )
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract all symbolic features from text.
        
        Args:
            text: User story description
            
        Returns:
            15-dimensional numpy array of features
        """
        if not text or len(text.strip()) == 0:
            return np.zeros(15)
        
        # Parse with spaCy
        doc = self.nlp(text)
        
        # Extract feature groups
        readability = self._extract_readability(text)
        ambiguity = self._extract_ambiguity(text, doc)
        lexicons = self._extract_lexicons(text)
        
        # Combine into single vector
        features = np.array([
            # Readability (3 features)
            readability['flesch_reading_ease'],
            readability['gunning_fog'],
            readability['lexical_density'],
            
            # Ambiguity (3 features)
            ambiguity['weak_modal_density'],
            ambiguity['has_vague_quantifiers'],
            ambiguity['passive_voice_ratio'],
            
            # Risk Lexicons (3 features)
            lexicons['satd_count'],
            lexicons['security_count'],
            lexicons['complexity_count'],
            
            # Text statistics (6 features)
            len(text),                           # Character count
            len(text.split()),                   # Word count
            len(list(doc.sents)),                # Sentence count
            self._count_questions(text),         # Question marks
            self._count_code_blocks(text),       # Code block markers
            self._count_list_items(text)         # Bullet points
        ], dtype=np.float32)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names in order."""
        return [
            'flesch_reading_ease',
            'gunning_fog',
            'lexical_density',
            'weak_modal_density',
            'has_vague_quantifiers',
            'passive_voice_ratio',
            'satd_count',
            'security_count',
            'complexity_count',
            'char_count',
            'word_count',
            'sentence_count',
            'question_count',
            'code_block_count',
            'list_item_count'
        ]
    
    def _extract_readability(self, text: str) -> Dict[str, float]:
        """
        Extract readability metrics.
        
        Returns:
            Dict with flesch_reading_ease, gunning_fog, lexical_density
        """
        try:
            flesch = textstat.flesch_reading_ease(text)
            fog = textstat.gunning_fog(text)
        except:
            # Handle edge cases (very short text)
            flesch = 100.0  # Easy to read
            fog = 0.0       # Low complexity
        
        # Calculate lexical density
        words = text.lower().split()
        if len(words) == 0:
            lexical_density = 0.0
        else:
            doc = self.nlp(text)
            content_words = sum(1 for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'])
            lexical_density = content_words / len(words) if len(words) > 0 else 0.0
        
        return {
            'flesch_reading_ease': flesch,
            'gunning_fog': fog,
            'lexical_density': lexical_density
        }
    
    def _extract_ambiguity(self, text: str, doc) -> Dict[str, float]:
        """
        Extract ambiguity indicators.
        
        Returns:
            Dict with weak_modal_density, has_vague_quantifiers, passive_voice_ratio
        """
        sentences = list(doc.sents)
        num_sentences = max(len(sentences), 1)
        
        # Weak modal density
        weak_modals = len(self._weak_modal_pattern.findall(text))
        modal_density = weak_modals / num_sentences
        
        # Vague quantifiers (binary flag)
        has_vague = 1.0 if self._vague_pattern.search(text) else 0.0
        
        # Passive voice ratio
        passive_sentences = sum(1 for sent in sentences if self._is_passive(sent))
        passive_ratio = passive_sentences / num_sentences if num_sentences > 0 else 0.0
        
        return {
            'weak_modal_density': modal_density,
            'has_vague_quantifiers': has_vague,
            'passive_voice_ratio': passive_ratio
        }
    
    def _extract_lexicons(self, text: str) -> Dict[str, int]:
        """
        Count occurrences of risk keywords.
        
        Returns:
            Dict with satd_count, security_count, complexity_count
        """
        text_lower = text.lower()
        
        satd_count = sum(text_lower.count(kw) for kw in self.SATD_KEYWORDS)
        security_count = sum(text_lower.count(kw) for kw in self.SECURITY_KEYWORDS)
        complexity_count = sum(text_lower.count(kw) for kw in self.COMPLEXITY_KEYWORDS)
        
        return {
            'satd_count': satd_count,
            'security_count': security_count,
            'complexity_count': complexity_count
        }
    
    def _is_passive(self, sentence) -> bool:
        """
        Detect if a sentence uses passive voice.
        
        Heuristic: Look for auxiliary verb + past participle
        """
        for token in sentence:
            if token.dep_ == "nsubjpass":  # Passive subject
                return True
        return False
    
    def _count_questions(self, text: str) -> int:
        """Count question marks."""
        return text.count('?')
    
    def _count_code_blocks(self, text: str) -> int:
        """Count code block markers (```)."""
        return text.count('```') // 2  # Pairs of markers
    
    def _count_list_items(self, text: str) -> int:
        """Count bullet points and list markers."""
        pattern = re.compile(r'[-*•]\s', re.MULTILINE)
        return len(pattern.findall(text))

