"""
Flitter generic controller framework
"""

import asyncio

from loguru import logger

from ...plugins import get_plugin
from . import midi


class Controller:
    VIRTUAL_MIDI_PORT = None
    DRIVER_CACHE = {}

    @classmethod
    def open_virtual_midi_port(cls):
        """Hold a virtual MIDI port open to stop RTMIDI on OS X releasing the
           main context when ports are closed and throwing subsequent errors
           when attempting to open a new port."""
        if cls.VIRTUAL_MIDI_PORT is None:
            cls.VIRTUAL_MIDI_PORT = midi.MidiPort('flitter', virtual=True)

    def __init__(self, **kwargs):
        logger.debug("Create controller")
        self.open_virtual_midi_port()
        self.driver = None
        self.controls = {}
        self.unknown = set()

    async def purge(self):
        while self.controls:
            key, control = self.controls.popitem()
            control.reset()
            control.update_representation()

    async def destroy(self):
        if self.driver is not None:
            await self.purge()
            await self.driver.stop()
            self.driver = None

    async def update(self, engine, node, time, **kwargs):
        driver = node.get('driver', 1, str)
        driver_class = get_plugin('flitter.render.controller', driver)
        if self.driver is not None and (driver_class is None or not isinstance(self.driver, driver_class)):
            await self.driver.stop()
            self.driver = None
        if self.driver is None:
            if driver_class is not None:
                self.driver = driver_class(engine)
                await self.driver.start()
            else:
                return
        await self.driver.start_update(node)
        controls = {}
        unknown = set()
        for child in list(node.children) + self.driver.get_default_config():
            if 'id' in child:
                control_id = child['id']
                control = self.driver.get_control(child.kind, control_id)
                if control is not None:
                    key = child.kind, control_id
                    if key in controls:
                        continue
                    controls[key] = control
                    if key in self.controls:
                        del self.controls[key]
                    if control.update(child, time):
                        control.update_representation()
                    control.update_state()
                    continue
            if not self.driver.handle_node(child):
                unknown.add(child.kind)
        for kind in unknown.difference(self.unknown):
            logger.warning("Unhandled node in controller: {!r}", child)
        await self.purge()
        self.unknown = unknown
        self.controls = controls
        await self.driver.finish_update()
        await asyncio.sleep(0)
