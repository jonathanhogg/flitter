"""
Flitter generic controller framework
"""

import asyncio
import importlib

from loguru import logger

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

    def purge(self):
        while self.controls:
            key, control = self.controls.popitem()
            control.reset()
            control.update_representation()

    def destroy(self):
        if self.driver is not None:
            self.purge()
            asyncio.create_task(self.driver.stop())
            self.driver = None

    async def update(self, engine, node, clock, **kwargs):
        driver = node.get('driver', 1, str)
        if driver in self.DRIVER_CACHE:
            driver_class = self.DRIVER_CACHE[driver]
        elif driver:
            try:
                driver_module = importlib.import_module(f'.{driver}', __package__)
                driver_class = driver_module.get_driver_class()
            except (ImportError, NameError):
                logger.warning("Unable to import controller driver: {}", driver)
                driver_class = None
            self.DRIVER_CACHE[driver] = driver_class
        else:
            driver_class = None
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
                    if control.update(child, clock):
                        control.update_representation()
                    control.update_state()
                    continue
            if not self.driver.handle_node(child):
                unknown.add(child.kind)
        for kind in unknown.difference(self.unknown):
            logger.warning("Unexpected '{}' node in controller", kind)
        self.purge()
        self.unknown = unknown
        self.controls = controls
        await self.driver.finish_update()
        await asyncio.sleep(0)


RENDERER_CLASS = Controller
