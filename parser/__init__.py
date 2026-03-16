from .lexer import tokenize, LexerError
from .parser import Parser, ParseError
from .emitter import emit


def parse_condition(text: str) -> dict:
    """
    Parse a RAVEN condition string and return a structured result.

    Returns
    -------
    dict with keys:
      "json"   – emitted AST as a JSON-serialisable dict
      "tokens" – list of token dicts (type, value, position)
    """
    tokens = tokenize(text)
    parser = Parser(tokens)
    ast = parser.parse()

    return {
        "json":   emit(ast),
        "tokens": [{"type": t.type, "value": t.value, "position": t.position}
                   for t in tokens],
    }