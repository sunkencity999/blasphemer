"""
Dataset processing for fine-tuning.
Handles PDF, text files, and HuggingFace datasets.
"""

import os
from pathlib import Path
from typing import List, Dict, Union, Optional
import PyPDF2
from datasets import Dataset, load_dataset
from rich import print


class DatasetProcessor:
    """Process various data sources into training-ready datasets."""
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        train_test_split: float = 0.9,
    ):
        """
        Initialize dataset processor.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between chunks for context
            train_test_split: Ratio of training data (0.9 = 90% train, 10% val)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.train_test_split = train_test_split
    
    def process(
        self,
        source: str,
        tokenizer,
    ) -> Dict[str, Dataset]:
        """
        Process data source into training datasets.
        
        Args:
            source: Path to file/directory or HuggingFace dataset name
            tokenizer: Tokenizer for chunking
        
        Returns:
            Dictionary with 'train' and 'validation' datasets
        """
        print(f"[cyan]Processing dataset: {source}[/]")
        
        # Determine source type
        path = Path(source)
        
        if path.exists():
            if path.is_file():
                texts = self._process_file(path)
            elif path.is_dir():
                texts = self._process_directory(path)
            else:
                raise ValueError(f"Unsupported source type: {source}")
        else:
            # Assume HuggingFace dataset
            texts = self._process_huggingface(source)
        
        print(f"[green]✓ Extracted {len(texts)} text segments[/]")
        
        # Chunk texts
        chunks = self._chunk_texts(texts, tokenizer)
        print(f"[green]✓ Created {len(chunks)} training chunks[/]")
        
        # Format as instruction-response pairs
        examples = self._create_examples(chunks)
        
        # Create dataset and split
        dataset = Dataset.from_dict({
            'instruction': [ex['instruction'] for ex in examples],
            'response': [ex['response'] for ex in examples],
        })
        
        # Split into train/validation
        split_dataset = dataset.train_test_split(
            train_size=self.train_test_split,
            seed=42,
        )
        
        print(f"[green]✓ Training examples: {len(split_dataset['train'])}[/]")
        print(f"[green]✓ Validation examples: {len(split_dataset['test'])}[/]")
        
        return {
            'train': split_dataset['train'],
            'validation': split_dataset['test'],
        }
    
    def _process_file(self, path: Path) -> List[str]:
        """Process a single file."""
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            return self._extract_from_pdf(path)
        elif suffix in ['.txt', '.md', '.markdown']:
            return self._extract_from_text(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def _process_directory(self, path: Path) -> List[str]:
        """Process all supported files in directory recursively."""
        texts = []
        
        # Supported extensions
        extensions = {'.pdf', '.txt', '.md', '.markdown'}
        
        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                print(f"  Processing: {file_path.name}")
                try:
                    texts.extend(self._process_file(file_path))
                except Exception as e:
                    print(f"  [yellow]⚠ Skipped {file_path.name}: {e}[/]")
        
        if not texts:
            raise ValueError(f"No supported files found in {path}")
        
        return texts
    
    def _extract_from_pdf(self, path: Path) -> List[str]:
        """Extract text from PDF file."""
        texts = []
        
        try:
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        texts.append(text.strip())
        
        except Exception as e:
            raise ValueError(f"Failed to read PDF {path.name}: {e}")
        
        return texts
    
    def _extract_from_text(self, path: Path) -> List[str]:
        """Extract text from plain text file."""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Split by paragraphs (double newline)
                paragraphs = [
                    p.strip() 
                    for p in content.split('\n\n')
                    if p.strip()
                ]
                
                return paragraphs if paragraphs else [content]
        
        except Exception as e:
            raise ValueError(f"Failed to read text file {path.name}: {e}")
    
    def _process_huggingface(self, dataset_name: str) -> List[str]:
        """Load and extract text from HuggingFace dataset."""
        try:
            # Load dataset
            dataset = load_dataset(dataset_name, split='train')
            
            # Try to find text column
            text_columns = ['text', 'content', 'document', 'article', 'passage']
            text_col = None
            
            for col in text_columns:
                if col in dataset.column_names:
                    text_col = col
                    break
            
            if not text_col:
                # Use first column if no standard text column found
                text_col = dataset.column_names[0]
                print(f"  [yellow]Using column '{text_col}' for text[/]")
            
            texts = [str(item[text_col]) for item in dataset]
            return texts
        
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace dataset {dataset_name}: {e}")
    
    def _chunk_texts(self, texts: List[str], tokenizer) -> List[str]:
        """
        Chunk texts into smaller segments.
        Uses sliding window with overlap for context preservation.
        """
        chunks = []
        
        for text in texts:
            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Skip very short texts
            if len(tokens) < 50:
                continue
            
            # If text fits in one chunk, use as-is
            if len(tokens) <= self.chunk_size:
                chunks.append(text)
                continue
            
            # Split into overlapping chunks
            start = 0
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                
                # Decode back to text
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
                
                # Move forward with overlap
                if end >= len(tokens):
                    break
                start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _create_examples(self, chunks: List[str]) -> List[Dict[str, str]]:
        """
        Create instruction-response pairs for knowledge injection.
        
        Uses various instruction templates to create diverse training data.
        """
        examples = []
        
        instruction_templates = [
            "What information do you have about the following topic?",
            "Can you explain what you know about this subject?",
            "Please provide information on this topic:",
            "What can you tell me about this?",
            "Explain the following in detail:",
            "Share what you know about:",
            "Provide a detailed explanation of:",
            "What do you understand about this topic?",
        ]
        
        for i, chunk in enumerate(chunks):
            # Cycle through instruction templates
            template = instruction_templates[i % len(instruction_templates)]
            
            # Extract first sentence or first 100 chars as topic
            topic = chunk.split('.')[0][:100]
            if len(topic) < 10:
                topic = chunk[:100]
            
            examples.append({
                'instruction': f"{template} {topic}",
                'response': chunk,
            })
        
        return examples
    
    def preview_examples(
        self,
        dataset: Dict[str, Dataset],
        num_examples: int = 3,
    ):
        """Preview training examples."""
        print("\n[bold cyan]Training Data Preview:[/]")
        print("=" * 80)
        
        for i in range(min(num_examples, len(dataset['train']))):
            example = dataset['train'][i]
            print(f"\n[bold]Example {i+1}:[/]")
            print(f"[yellow]Instruction:[/] {example['instruction'][:150]}...")
            print(f"[green]Response:[/] {example['response'][:300]}...")
            print("-" * 80)
