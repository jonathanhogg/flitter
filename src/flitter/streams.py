"""
streams
=======

Package for asynchronous serial IO.
"""

import asyncio

from loguru import logger
import serial
from serial.tools.list_ports import comports
from serial.serialutil import SerialException


class SerialStream:

    @classmethod
    def devices_matching(cls, vid=None, pid=None, serial_number=None):
        for port in comports():
            if (vid is None or vid == port.vid) and (pid is None or pid == port.pid) and (serial_number is None or serial_number == port.serial_number):
                yield port.device

    @classmethod
    def stream_matching(cls, vid=None, pid=None, serial_number=None, **kwargs):
        for device in cls.devices_matching(vid, pid, serial_number):
            return SerialStream(device, **kwargs)
        raise SerialException("No matching serial device")

    def __init__(self, device, **kwargs):
        self._device = device
        self._connection = serial.Serial(self._device, timeout=0, write_timeout=0, **kwargs)
        logger.debug("Opened SerialStream on {} ({!r})", device, ', '.join(f'{k}={v!r}' for k, v in kwargs.items()))
        self._loop = asyncio.get_event_loop()
        self._output_buffer = bytes()
        self._output_buffer_empty = None

    def __repr__(self):
        return f'<{self.__class__.__name__}:{self._device}>'

    @property
    def dtr(self):
        return self._connection.dtr

    @dtr.setter
    def dtr(self, value):
        self._connection.dtr = bool(value)

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def write(self, data):
        if not self._output_buffer:
            try:
                nbytes = self._connection.write(data)
            except serial.SerialTimeoutException:
                nbytes = 0
            except Exception:
                logger.exception("Error writing to stream")
                raise
            if nbytes:
                logger.trace("Write {!r}", data[:nbytes])
            self._output_buffer = data[nbytes:]
        else:
            self._output_buffer += data
        if self._output_buffer and self._output_buffer_empty is None:
            self._output_buffer_empty = self._loop.create_future()
            self._loop.add_writer(self._connection, self._feed_data)

    async def drain(self):
        if self._output_buffer_empty is not None:
            await self._output_buffer_empty

    def _feed_data(self):
        try:
            nbytes = self._connection.write(self._output_buffer)
        except serial.SerialTimeoutException:
            nbytes = 0
        except Exception as exc:
            logger.exception("Error writing to stream")
            self._output_buffer_empty.set_exception(exc)
            self._loop.remove_writer(self._connection)
        if nbytes:
            logger.trace("Write {!r}", self._output_buffer[:nbytes])
            self._output_buffer = self._output_buffer[nbytes:]
        if not self._output_buffer:
            self._loop.remove_writer(self._connection)
            self._output_buffer_empty.set_result(None)
            self._output_buffer_empty = None

    async def read(self, nbytes=None):
        while True:
            nwaiting = self._connection.in_waiting
            if nwaiting:
                data = self._connection.read(nwaiting if nbytes is None else min(nbytes, nwaiting))
                logger.trace("Read {!r}", data)
                return data
            future = self._loop.create_future()
            self._loop.add_reader(self._connection, future.set_result, None)
            try:
                await future
            finally:
                self._loop.remove_reader(self._connection)

    async def readexactly(self, nbytes):
        data = b''
        while len(data) < nbytes:
            data += await self.read(nbytes - len(data))
        return data
