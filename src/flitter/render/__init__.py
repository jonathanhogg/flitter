"""
Flitter generic rendering
"""


class Renderer:
    """
    Renderers should implement this API
    """
    def __init__(self, **kwargs):
        """
        `kwargs` will contain the engine global definitions. You will likely
        want to create most of the renderer state on first call of `update()`
        when you have access to the top-level node attributes.
        """
        pass

    async def purge(self):
        """
        `purge()`` should wipe any local state ready for a complete redefinition.
        """
        raise NotImplementedError()

    async def destroy(self):
        """
        `destroy()`` should tear down the renderer itself.
        """
        raise NotImplementedError()

    async def update(self, engine, node, **kwargs):
        """
        `update()`` is called every frame with the node to render and the engine
        global definitions (including `time` and `beat`).
        """
        raise NotImplementedError()
