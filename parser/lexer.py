import re
from dataclasses import dataclass


class LexerError(Exception):
    def __init__(self, message: str, position: int = -1):
        super().__init__(message)
        self.position = position


@dataclass
class Token:
    type: str
    value: str
    position: int


LOGICAL = {"and", "or", "not"}

# Compile all patterns once at module load for performance
_TOKEN_SPECS = [
    ("LPAREN",     re.compile(r"\(")),
    ("RPAREN",     re.compile(r"\)")),
    ("COMMA",      re.compile(r",")),
    ("NUMBER",     re.compile(r"-?\d+(\.\d+)?")),
    ("STRING",     re.compile(r'"[^"]*"|\'[^\']*\'')),
    ("OP",         re.compile(r"!=|>=|<=|>|<|=")),
    ("WORD",       re.compile(r"[A-Za-z_][A-Za-z0-9_]*")),
]


def tokenize(text: str) -> list:
    tokens = []
    pos = 0
    length = len(text)

    while pos < length:

        if text[pos].isspace():
            pos += 1
            continue

        matched = False
        for name, regex in _TOKEN_SPECS:
            m = regex.match(text, pos)
            if not m:
                continue

            val = m.group(0)

            if name == "WORD":
                if val.lower() in LOGICAL:
                    tokens.append(Token("LOGICAL", val.upper(), pos))
                else:
                    tokens.append(Token("FIELD", val, pos))

            elif name == "NUMBER":
                tokens.append(Token("VALUE_NUM", val, pos))

            elif name == "STRING":
                # Strip surrounding quotes and store clean value
                tokens.append(Token("VALUE_STR", val[1:-1], pos))

            elif name == "OP":
                tokens.append(Token("OPERATOR", val, pos))

            else:
                tokens.append(Token(name, val, pos))

            pos = m.end()
            matched = True
            break

        if not matched:
            raise LexerError(
                f"Unexpected character '{text[pos]}' at position {pos}",
                position=pos,
            )

    return tokens