"""
1. divide corpus words into chars -> initial vocab
 - Words which do not start words have ## at the beginning
2. Compute pair scores 
 - freq of pair / (freq of 1st element  * freq of 2nd element)
3. Add the element with highest score to the corpus
4. repeat step 2 and 3 until satisfied 
"""

class WordPieceEncoder:
    def __init__(self, data):
        self.data = data
    
    def _build_vocab(self):
        corpus = self.corpus.split(" ")

 

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
