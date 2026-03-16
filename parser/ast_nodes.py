from dataclasses import dataclass, field
from typing import Union, Optional


@dataclass
class RuleNode:
    field: str
    operator: str
    value: Optional[object]
    value_type: str
    position: int = 0          # character offset in source text


@dataclass
class ListNode:
    """Represents  field IN (val1, val2, ...)"""
    field: str
    operator: str              # always "in" or "not in"
    values: list
    value_type: str            # "Number" | "Text" | "Mixed"
    position: int = 0


@dataclass
class LogicalNode:
    logical_operator: str      # "AND" | "OR"
    left: "AstNode"
    right: "AstNode"


@dataclass
class NotNode:
    inner: "AstNode"


AstNode = Union[RuleNode, ListNode, LogicalNode, NotNode]