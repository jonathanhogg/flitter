"""
The main Flitter engine
"""

from pathlib import Path

from flitter.clock import BeatCounter
from flitter.language.interpreter import simplify, evaluate
from flitter.language.parser import parse
from flitter.model import Context, Vector, Node, null
from flitter.render.scene import Window


class Controller:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.state = {}
        self.tree = None
        self.simplified = None
        self.windows = []
        self.counter = BeatCounter()

    def load(self, filename):
        with open(filename, encoding='utf8') as file, Context() as context:
            self.tree = simplify(parse(file.read()), context)
            self.simplified = None

    def set_state(self, key, value):
        if key not in self.state or value != self.state[key]:
            self.state[key] = value
            self.simplified = None

    def read(self, filename):
        if len(filename) == 1 and filename.isinstance(str):
            with open(self.root_dir / filename[0], encoding='utf8') as file:
                return Vector((file.read(),))
        return null

    def execute(self):
        if self.simplified is None:
            with Context(state=self.state) as context:
                self.simplified = simplify(self.tree, context)
        variables = {'beat': Vector((self.counter.beat,)), 'read': Vector((self.read,))}
        with Context(variables=variables, state=self.state) as context:
            for expr in self.simplified.expressions:
                result = evaluate(expr, context)
                for value in result:
                    if isinstance(value, Node) and value.parent is None:
                        context.graph.append(value)
        return context.graph

    def update_windows(self, graph):
        count = 0
        for i, node in enumerate(graph.select_below('window.')):
            if i == len(self.windows):
                self.windows.append(Window())
            self.windows[i].update(node)
            count += 1
        while len(self.windows) > count:
            self.windows.pop().destroy()
