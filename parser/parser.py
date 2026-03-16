from .ast_nodes import RuleNode, ListNode, LogicalNode, NotNode


class ParseError(Exception):
    def __init__(self, message: str, position: int = -1):
        super().__init__(message)
        self.position = position


class Parser:

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def expect(self, type_: str):
        tok = self.consume()
        if tok is None:
            raise ParseError(f"Expected {type_!r} but reached end of input")
        if tok.type != type_:
            raise ParseError(
                f"Expected token type {type_!r} but got {tok.type!r} ({tok.value!r})",
                position=tok.position,
            )
        return tok

    # ------------------------------------------------------------------
    # Grammar
    # ------------------------------------------------------------------

    def parse(self):
        node = self.parse_expression()
        if self.peek() is not None:
            tok = self.peek()
            raise ParseError(
                f"Unexpected token {tok.value!r} at position {tok.position}",
                position=tok.position,
            )
        return node

    def parse_expression(self):
        node = self.parse_and()
        while self.peek() and self.peek().value == "OR":
            self.consume()
            right = self.parse_and()
            node = LogicalNode("OR", node, right)
        return node

    def parse_and(self):
        node = self.parse_not()
        while self.peek() and self.peek().value == "AND":
            self.consume()
            right = self.parse_not()
            node = LogicalNode("AND", node, right)
        return node

    def parse_not(self):
        tok = self.peek()
        if tok and tok.value == "NOT":
            self.consume()
            inner = self.parse_group()
            return NotNode(inner)
        return self.parse_group()

    def parse_group(self):
        tok = self.peek()
        if tok and tok.type == "LPAREN":
            self.consume()
            node = self.parse_expression()
            if not self.peek() or self.peek().type != "RPAREN":
                pos = self.peek().position if self.peek() else -1
                raise ParseError("Missing closing parenthesis ')'", position=pos)
            self.consume()
            return node
        return self.parse_rule()

    def parse_rule(self):
        field_tok = self.peek()
        if field_tok is None or field_tok.type != "FIELD":
            pos = field_tok.position if field_tok else -1
            raise ParseError(
                f"Expected a field name but got {field_tok.value!r}" if field_tok
                else "Expected a field name but reached end of input",
                position=pos,
            )
        field_tok = self.consume()

        op_tok = self.peek()
        if op_tok is None or op_tok.type != "OPERATOR":
            pos = op_tok.position if op_tok else -1
            raise ParseError(
                f"Expected an operator after '{field_tok.value}' but got "
                + (f"{op_tok.value!r}" if op_tok else "end of input"),
                position=pos,
            )
        op_tok = self.consume()

        # Handle list literal:  field = (val1, val2, ...)
        if self.peek() and self.peek().type == "LPAREN":
            return self._parse_list(field_tok, op_tok)

        val_tok = self.peek()
        if val_tok is None:
            raise ParseError(
                f"Expected a value after '{field_tok.value} {op_tok.value}' "
                "but reached end of input",
                position=op_tok.position,
            )
        if val_tok.type not in ("VALUE_NUM", "VALUE_STR", "FIELD"):
            raise ParseError(
                f"Expected a value but got {val_tok.type!r} ({val_tok.value!r})",
                position=val_tok.position,
            )
        val_tok = self.consume()

        if val_tok.type == "VALUE_NUM":
            return RuleNode(
                field=field_tok.value,
                operator=op_tok.value,
                value=float(val_tok.value),
                value_type="Number",
                position=field_tok.position,
            )

        # VALUE_STR (quoted) or FIELD used as a bare text value
        return RuleNode(
            field=field_tok.value,
            operator=op_tok.value,
            value=val_tok.value,
            value_type="Text",
            position=field_tok.position,
        )

    def _parse_list(self, field_tok, op_tok):
        """Parse  (val1, val2, ...)  after the operator."""
        self.consume()  # consume LPAREN
        values = []
        value_types = set()

        while True:
            val_tok = self.peek()
            if val_tok is None:
                raise ParseError("Unclosed list — missing ')'", position=op_tok.position)
            if val_tok.type == "RPAREN":
                self.consume()
                break
            if val_tok.type == "COMMA":
                self.consume()
                continue
            if val_tok.type not in ("VALUE_NUM", "VALUE_STR", "FIELD"):
                raise ParseError(
                    f"Unexpected token inside list: {val_tok.value!r}",
                    position=val_tok.position,
                )
            self.consume()
            if val_tok.type == "VALUE_NUM":
                values.append(float(val_tok.value))
                value_types.add("Number")
            else:
                values.append(val_tok.value)
                value_types.add("Text")

        if not values:
            raise ParseError("Empty list is not allowed", position=op_tok.position)

        if len(value_types) == 1:
            vtype = value_types.pop()
        else:
            vtype = "Mixed"

        return ListNode(
            field=field_tok.value,
            operator=op_tok.value,
            values=values,
            value_type=vtype,
            position=field_tok.position,
        )