from .ast_nodes import RuleNode, ListNode, LogicalNode, NotNode


def emit(node) -> dict:
    """Recursively convert an AST node to a plain JSON-serialisable dict."""

    if isinstance(node, RuleNode):
        return {
            "field":     node.field,
            "operator":  node.operator,
            "value":     node.value,
            "valueType": node.value_type,
            "position":  node.position,
        }

    if isinstance(node, ListNode):
        return {
            "field":     node.field,
            "operator":  node.operator,
            "values":    node.values,
            "valueType": node.value_type,
            "position":  node.position,
        }

    if isinstance(node, NotNode):
        return {
            "logicalOperator": "NOT",
            "groups":          [emit(node.inner)],
            "rules":           [],
        }

    if isinstance(node, LogicalNode):
        left  = emit(node.left)
        right = emit(node.right)
        rules  = []
        groups = []
        for child in [left, right]:
            if "field" in child:
                rules.append(child)
            else:
                groups.append(child)
        return {
            "logicalOperator": node.logical_operator,
            "rules":           rules,
            "groups":          groups,
        }

    raise TypeError(f"Unknown AST node type: {type(node)}")