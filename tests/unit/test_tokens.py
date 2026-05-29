from reliquary.protocol.tokens import encode_prompt, int_to_bytes, hash_tokens


class TestIntToBytes:
    def test_zero(self):
        assert int_to_bytes(0) == b"\x00\x00\x00\x00"

    def test_one(self):
        assert int_to_bytes(1) == b"\x00\x00\x00\x01"

    def test_big_endian(self):
        assert int_to_bytes(256) == b"\x00\x00\x01\x00"

    def test_large_value(self):
        result = int_to_bytes(0xFFFFFFFF)
        assert result == b"\xff\xff\xff\xff"

    def test_always_4_bytes(self):
        for val in [0, 1, 255, 65535, 2**32 - 1]:
            assert len(int_to_bytes(val)) == 4


class TestHashTokens:
    def test_deterministic(self):
        tokens = [1, 2, 3, 4, 5]
        assert hash_tokens(tokens) == hash_tokens(tokens)

    def test_32_bytes(self):
        assert len(hash_tokens([1, 2, 3])) == 32

    def test_different_tokens_differ(self):
        assert hash_tokens([1, 2, 3]) != hash_tokens([3, 2, 1])

    def test_order_matters(self):
        assert hash_tokens([1, 2]) != hash_tokens([2, 1])

    def test_empty(self):
        result = hash_tokens([])
        assert len(result) == 32


class _ChatTokenizer:
    """Minimal stand-in for an instruct-model tokenizer.

    Mirrors the surface area encode_prompt depends on: a non-empty
    ``chat_template`` string and an ``apply_chat_template`` method that
    returns the chat-formatted token list. ``encode`` is also provided so
    the fallback path is observable in a test.
    """

    chat_template = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is True
        prompt = messages[0]["content"]
        return [1] + list(prompt.encode("utf-8"))

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


class _BareTokenizer:
    chat_template = None

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


class TestEncodePrompt:
    def test_uses_chat_template_when_present(self):
        tok = _ChatTokenizer()
        out = encode_prompt(tok, "hi")
        assert out == [1] + list(b"hi")

    def test_falls_back_when_no_chat_template(self):
        tok = _BareTokenizer()
        assert encode_prompt(tok, "hi") == list(b"hi")

    def test_miner_and_validator_agree(self):
        """The single shared helper guarantees byte-equal prompt tokens on
        both sides, so the validator's PROMPT_MISMATCH gate cannot trip on
        an honest miner that went through the same path."""
        tok = _ChatTokenizer()
        prompt = "Solve 2 + 2"
        miner_tokens = encode_prompt(tok, prompt)
        validator_tokens = encode_prompt(tok, prompt)
        assert miner_tokens == validator_tokens

    def test_magicmock_tokenizer_falls_back(self):
        """MagicMock returns truthy values for any attribute access; the
        helper must still fall through to plain encode() so existing test
        stubs continue to work."""
        from unittest.mock import MagicMock

        tok = MagicMock()
        tok.encode.return_value = [7, 8, 9]
        out = encode_prompt(tok, "anything")
        assert out == [7, 8, 9]
        tok.apply_chat_template.assert_not_called()
