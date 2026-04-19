import pandas as pd
import numpy as np
import typing
from collections import deque, defaultdict


class Tokenizer(): 
    def __init__(self): 

        self.symbol_set : set = None
        self.symbol_to_token = {}
        self.token_to_symbol = {}
        self.language_size = 0
        self.corpus = None
    def train_tokenizer(self, input, max_language_size: int) -> None:
        if type(input) == str:
            self.corpus = input.split(",")
        else:
            self.corpus = input
        self.symbol_set = set(self.corpus)
        for sym in self.symbol_set:
            self.symbol_to_token[sym] = self.language_size
            self.token_to_symbol[self.language_size] = sym
            self.language_size += 1
        # Converted everythign to tokens from symbolic form
        self.corpus = np.array([self.symbol_to_token[sym] for sym in self.corpus], dtype=int)
        
        while self.language_size < max_language_size:
            temp_corpus = self.corpus
            common_pair = None
            highest_pair_count = 0
            pair_counts = defaultdict(int)
            for i in range(len(temp_corpus)-1):
                pair = (temp_corpus[i], temp_corpus[i+1])
                pair_counts[pair] += 1
                if (pair_counts[pair] > highest_pair_count):
                    highest_pair_count = pair_counts[pair]
                    common_pair = pair
            synthetic_symbol = self.token_to_symbol[common_pair[0]] + self.token_to_symbol[common_pair[1]]

            self.symbol_to_token[synthetic_symbol] = self.language_size
            self.token_to_symbol[self.language_size] = synthetic_symbol

            self.language_size += 1
            combine_tokens = deque(temp_corpus) 
            self.corpus = []           

            while (len(combine_tokens) > 1):
                first_elem = combine_tokens.popleft()
                second_elem = combine_tokens.popleft()
                
                if ((first_elem, second_elem) == common_pair):
                    combine_tokens.appendleft(self.language_size - 1)
                
                else:
                    self.corpus.append(first_elem)
                    self.corpus.append(second_elem)
            if (len(combine_tokens) > 0):
                self.corpus.append(combine_tokens.popleft())

        self.corpus = None

    def decode(self, tokens: list[int]) -> str:
        return "".join([self.token_to_symbol[t] for t in tokens])

    def encode(self, message: str):
        char_list = list(message)
        char_inputs = deque(char_list)

        result_tokens = []
        curr_symbol = ""
        while (len(char_inputs) > 0):
            f_char = char_inputs.popleft()
            curr_symbol += f_char 

            if (curr_symbol not in self.symbol_to_token.keys()):
                curr_symbol = curr_symbol[:-1]
                result_tokens.append(self.symbol_to_token[curr_symbol])
                char_inputs.appendleft(f_char)
                curr_symbol = ""
        if (len(curr_symbol) > 0):
            result_tokens.append(self.symbol_to_token[curr_symbol])

        return result_tokens

    def encode_moves(self, moves: list[str]) -> list[int]:
        return [self.symbol_to_token[move] for move in moves]

    def add_special_tokens(self, tokens: list[str]) -> dict[str, int]:
        mapping = {}
        for tok in tokens:
            self.symbol_to_token[tok] = self.language_size
            self.token_to_symbol[self.language_size] = tok
            mapping[tok] = self.language_size
            self.language_size += 1
        return mapping


class DataLoader():
    corpus = None
    def __init__(self, file_name: str):
        with open(file_name, "r") as f:
            self.corpus = f.read()

