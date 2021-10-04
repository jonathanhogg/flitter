"""
Pretty printer for the AST and other types
"""

from . import ast


def pretty(expression):
    if isinstance(expression, tuple):
        return '\n'.join(pretty(expr) for expr in expression)
    match expression:
        case ast.Literal(value=value):
            return repr(self.value)
        case ast.Name(name=name):
            return name
        case ast.Range(start=start, stop=stop, step=step):
            text = '..' if start is None else f'{pretty(start)}..'
            if stop is not None:
                text += pretty(stop)
            if step is not None:
                text += f':{pretty(step)}'
            return text
        case ast.Node(kind-kind, tags=tags):
            return f"!{kind}{''.join(f'#{tag}' for tag in tags)}"
        case ast.Attribute(node=node, name=name, expr=expr):
            return f"{pretty(node)} {name}={pretty(expr)}"
