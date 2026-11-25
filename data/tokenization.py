import json
from pathlib import Path


def load_metadata(data_dir):
    with (Path(data_dir) / "meta.json").open() as file:
        return json.load(file)


def encode_text(text, metadata):
    if metadata["tokenizer"] == "char":
        stoi = {char: index for index, char in enumerate(metadata["chars"])}
        try:
            return [stoi[char] for char in text]
        except KeyError as error:
            raise ValueError(f"Character {error.args[0]!r} is not in the vocabulary") from error
    if metadata["tokenizer"] == "tiktoken":
        import tiktoken

        tokenizer = tiktoken.get_encoding(metadata["encoding"])
        return tokenizer.encode(text)
    raise ValueError(f"Unknown tokenizer: {metadata['tokenizer']}")


def decode_tokens(tokens, metadata):
    if metadata["tokenizer"] == "char":
        return "".join(metadata["chars"][index] for index in tokens)
    if metadata["tokenizer"] == "tiktoken":
        import tiktoken

        tokenizer = tiktoken.get_encoding(metadata["encoding"])
        return tokenizer.decode(tokens)
    raise ValueError(f"Unknown tokenizer: {metadata['tokenizer']}")
