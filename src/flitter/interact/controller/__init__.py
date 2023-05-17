"""
Flitter generic controller framework
"""

import importlib

from loguru import logger


class Controller:
    def __init__(self):
        logger.debug("Create controller")
        self.driver = None
        self.controls = {}

    def purge(self):
        while self.controls:
            key, control = self.controls.popitem()
            control.reset()
            control.update_representation()

    def destroy(self):
        if self.driver is not None:
            self.purge()
            self.driver.stop()
            self.driver = None

    async def update(self, engine, node, now):
        driver = node.get('driver', 1, str)
        driver_class = None
        if driver:
            try:
                driver_module = importlib.import_module(f'.{driver}', __package__)
                driver_class = driver_module.get_driver_class()
            except (ImportError, NameError):
                pass
        if self.driver is not None and (driver_class is None or not isinstance(self.driver, driver_class)):
            self.driver.stop()
            self.driver = None
        if self.driver is None:
            if driver_class is not None:
                self.driver = driver_class(node)
                await self.driver.start()
            else:
                return
        controls = {}
        for child in list(node.children) + self.driver.DEFAULT_CONFIG:
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
                    if control.update(engine, child, now):
                        control.update_representation()
        self.purge()
        self.controls = controls


INTERACTOR_CLASS = Controller
