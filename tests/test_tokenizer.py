import pytest
import numpy as np
from tokenizer import Tokenizer


class TestTokenizerInit:
    def test_initial_symbol_set_is_none(self):
        t = Tokenizer()
        assert t.symbol_set is None

    def test_initial_symbol_to_token_is_empty(self):
        t = Tokenizer()
        assert t.symbol_to_token == {}

    def test_initial_token_to_symbol_is_empty(self):
        t = Tokenizer()
        assert t.token_to_symbol == {}

    def test_initial_language_size_is_zero(self):
        t = Tokenizer()
        assert t.language_size == 0

    def test_initial_corpus_is_none(self):
        t = Tokenizer()
        assert t.corpus is None


class TestTrainTokenizer:
    def test_string_input_splits_on_comma(self):
        t = Tokenizer()
        t.train_tokenizer("a,b,c", max_language_size=3)
        assert set(t.symbol_set) == {"a", "b", "c"}

    def test_list_input_stored_directly(self):
        t = Tokenizer()
        t.train_tokenizer(["a", "b", "c"], max_language_size=3)
        assert set(t.symbol_set) == {"a", "b", "c"}

    def test_builds_symbol_to_token_mapping(self):
        t = Tokenizer()
        t.train_tokenizer(["a", "b"], max_language_size=2)
        assert "a" in t.symbol_to_token
        assert "b" in t.symbol_to_token

    def test_builds_token_to_symbol_mapping(self):
        t = Tokenizer()
        t.train_tokenizer(["a", "b"], max_language_size=2)
        assert len(t.token_to_symbol) == 2
        symbols = set(t.token_to_symbol.values())
        assert symbols == {"a", "b"}

    def test_language_size_equals_unique_symbols(self):
        t = Tokenizer()
        t.train_tokenizer(["a", "b", "a", "c"], max_language_size=3)
        assert t.language_size == 3

    def test_corpus_converted_to_numpy_int_array(self):
        t = Tokenizer()
        t.train_tokenizer(["a", "b", "c"], max_language_size=3)
        assert isinstance(t.corpus, np.ndarray)
        assert t.corpus.dtype == int

    def test_mappings_are_consistent(self):
        t = Tokenizer()
        t.train_tokenizer(["x", "y", "z"], max_language_size=3)
        for sym, tok in t.symbol_to_token.items():
            assert t.token_to_symbol[tok] == sym


class TestDecode:
    def test_decodes_tokens_to_string(self):
        t = Tokenizer()
        t.train_tokenizer(["a", "b", "c"], max_language_size=3)
        token_a = t.symbol_to_token["a"]
        token_b = t.symbol_to_token["b"]
        token_c = t.symbol_to_token["c"]
        result = t.decode([token_a, token_b, token_c])
        assert result == "abc"

    def test_decodes_empty_list(self):
        t = Tokenizer()
        t.token_to_symbol = {}
        result = t.decode([])
        assert result == ""

    def test_decodes_single_token(self):
        t = Tokenizer()
        t.train_tokenizer(["x"], max_language_size=1)
        token_x = t.symbol_to_token["x"]
        assert t.decode([token_x]) == "x"

    def test_decodes_repeated_tokens(self):
        t = Tokenizer()
        t.train_tokenizer(["a", "b"], max_language_size=2)
        token_a = t.symbol_to_token["a"]
        token_b = t.symbol_to_token["b"]
        assert t.decode([token_a, token_a, token_b, token_b]) == "aabb"

    def test_missing_token_raises_key_error(self):
        t = Tokenizer()
        t.train_tokenizer(["a"], max_language_size=1)
        with pytest.raises(KeyError):
            t.decode([99])


class TestEncode:
    def test_encode_single_characters(self):
        t = Tokenizer()
        t.train_tokenizer(["h", "i"], max_language_size=2)
        tokens = t.encode("hi")
        assert isinstance(tokens, list)
        assert len(tokens) == 2
        assert tokens == [t.symbol_to_token["h"], t.symbol_to_token["i"]]

    def test_encode_returns_list_of_ints(self):
        t = Tokenizer()
        t.train_tokenizer(["a", "b", "c"], max_language_size=3)
        tokens = t.encode("abc")
        assert all(isinstance(tok, int) for tok in tokens)

    def test_encode_empty_string(self):
        t = Tokenizer()
        t.train_tokenizer(["a"], max_language_size=1)
        tokens = t.encode("")
        assert tokens == []

    def test_encode_unknown_char_raises_key_error(self):
        t = Tokenizer()
        t.train_tokenizer(["a", "b"], max_language_size=2)
        with pytest.raises(KeyError):
            t.encode("z")



long_passage = '''
The Social Democrats, led by Prime Minister Mette Frederiksen (pictured), remain the largest party after the Danish general election, with no political bloc winning a majority of seats.
In Italy, voters reject a reform of the judicial system in a constitutional referendum.
A Colombian Aerospace Force Lockheed C-130 crashes during take-off in Puerto Leguízamo, killing 70 people.
In mathematics, Gerd Faltings is awarded the Abel Prize for his work in arithmetic geometry.
'''.lower()
class TestEncodeDecodeRoundTrip:
    def test_decode_of_encode_returns_original(self):
        t = Tokenizer()
        t.train_tokenizer(["a", "b", "c"], max_language_size=3)
        original = "abcabc"
        tokens = t.encode(original)
        decoded = t.decode(tokens)
        assert decoded == original

    def test_roundtrip_single_char(self):
        t = Tokenizer()
        t.train_tokenizer(["x"], max_language_size=1)
        assert t.decode(t.encode("x")) == "x"

    def test_roundtrip_repeated_chars(self):
        t = Tokenizer()
        t.train_tokenizer(["a", "b"], max_language_size=2)
        original = "aabba"
        assert t.decode(t.encode(original)) == original
    

    def test_encoder_from_long_passage(self):
        t = Tokenizer()
        t.train_tokenizer(list(long_passage), max_language_size=35)
        msg = "Gargzilla has IPS and MM happened during Spring Break".lower()
        print(t.encode(msg))
        assert t.decode(t.encode(msg)) == msg

