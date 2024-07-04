"""
Pixienet renderer
"""

import asyncio
import enum
import numpy as np
import struct
import socket
import time
import zlib

from loguru import logger


class FrameType(enum.IntEnum):
    QUERY = 0x01
    DATA = 0x02
    RESPONSE = 0x03


class LEDNetwork:
    DEVICE_PORT = 12346
    CONTROLLER_PORT = 12347

    class Device:
        def __init__(self, network, unique_id, format, address, port, bus_length, frame_count, version, timeout_wait=60):
            self.network = network
            self.unique_id = unique_id
            self.format = format
            self.address = address
            self.port = port
            self.bus_length = bus_length
            self.last_frame_count = frame_count
            self.version = version
            self.timeout_wait = timeout_wait
            self.led_count = self.bus_length // len(self.format)
            self.data = np.zeros((self.led_count, len(self.format)), dtype='uint8')
            self.sent_frames = 0
            self.last_poll_t = time.monotonic()
            self.dirty = asyncio.Event()
            logger.debug(f"LED network device {self.unique_id} (firmware {self.version}) located at {self.address} with {self.led_count} {self.format} LEDs")
            self._show_task = self.network._loop.create_task(self._show())

        def update(self, address, port, bus_length, frame_count, version, rssi):
            now = time.monotonic()
            if address != self.address:
                self.address = address
                logger.debug(f"Device {self.unique_id} changed address to {self.address}")
            if port != self.port:
                self.port = port
                logger.warning(f"Device {self.unique_id} changed port to {self.port}")
            if bus_length != self.bus_length:
                self.bus_length = bus_length
                self.led_count = self.bus_length // len(self.format)
                self.data = np.zeros((self.led_count, len(self.format)), dtype='uint8')
                logger.warning(f"Device {self.unique_id} changed LED count to {self.led_count}")
            if version != self.version:
                self.version = version
                logger.debug(f"Device {self.unique_id} changed firmware version to {self.version}")
            received_frames = frame_count - self.last_frame_count
            if received_frames < 0:
                logger.debug(f"Device {self.unique_id} reset")
            else:
                if received_frames < self.sent_frames:
                    logger.warning(f"Device {self.unique_id} dropped {self.sent_frames - received_frames} frames")
                logger.debug(f"Device {self.unique_id} processing frames at {received_frames / (now - self.last_poll_t):.1f}fps (RSSI {rssi}dB)")
            self.last_frame_count = frame_count
            self.sent_frames = 0
            self.last_poll_t = now
            if self._show_task.done():
                logger.debug(f"Restarting show task for device {self.unique_id}")
                self._show_task = self.network._loop.create_task(self._show())

        def __len__(self):
            return self.led_count

        def __setitem__(self, i, color):
            color = np.transpose(color)
            if color.shape[0] == 3 and len(self.format) == 4:
                w = color.min(axis=0)
                color = {'R': color[0] - w, 'G': color[1] - w, 'B': color[2] - w, 'W': w}
            elif color.shape[0] == 4 and len(self.format) == 3:
                w = color[3]
                color = {'R': color[0] + w, 'G': color[1] + w, 'B': color[2] + w}
            else:
                color = {'R': color[0], 'G': color[1], 'B': color[2], 'W': color[3] if color.shape[0] == 4 else 0}
            for j, c in enumerate(self.format):
                self.data[i, j] = np.clip(color[c] * 256, 0, 255)
            self.dirty.set()

        async def _show(self, minimum_wait=1/100, maximum_wait=1/5):
            try:
                frame_t = time.monotonic()
                while True:
                    frame = bytes([FrameType.DATA]) + zlib.compress(self.data.data)
                    self.dirty.clear()
                    self.network._socket.sendto(frame, (self.address, self.port))
                    self.sent_frames += 1
                    next_t = frame_t + max(minimum_wait, 8e-3 + self.led_count * len(self.format) / 53e3)
                    await asyncio.sleep(next_t - time.monotonic())
                    if self.dirty.is_set():
                        frame_t = next_t
                    else:
                        next_t = frame_t + maximum_wait
                        try:
                            await asyncio.wait_for(self.dirty.wait(), timeout=next_t - time.monotonic())
                            frame_t = time.monotonic()
                        except asyncio.TimeoutError:
                            frame_t = next_t
                    if time.monotonic() > self.last_poll_t + self.timeout_wait:
                        logger.error(f"No response from device {self.unique_id} in {self.timeout_wait} seconds; stopping show task")
                        return
            except Exception:
                logger.exception(f"Unexpected error in show task for device {self.unique_id}")
                raise

    def __init__(self, address='', DEVICE_PORT=DEVICE_PORT, CONTROLLER_PORT=CONTROLLER_PORT, format='GRBW', loop=None):
        self.DEVICE_PORT = DEVICE_PORT
        self.CONTROLLER_PORT = CONTROLLER_PORT
        self.format = format.upper()
        self._loop = asyncio.get_event_loop() if loop is None else loop
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._socket.bind((address, self.CONTROLLER_PORT))
        self._loop.add_reader(self._socket, self._packet_ready)
        self._poll_task = self._loop.create_task(self._poll())
        self._devices = {}
        logger.debug("Starting pixel network on {}:{}", address, self.CONTROLLER_PORT)

    def close(self):
        logger.debug("Closing pixel network")
        if self._socket is not None:
            self._loop.remove_reader(self._socket)
            self._socket.close()
            self._socket = None
        if self._poll_task is not None:
            self._poll_task.cancel()
            self._poll_task = None
        for device in self._devices.values():
            device.network = None
            if not device._show_task.done():
                device._show_task.cancel()
        self._devices.clear()

    async def _poll(self, period=10):
        try:
            while True:
                self._socket.sendto(bytes([FrameType.QUERY]), ('<broadcast>', self.DEVICE_PORT))
                await asyncio.sleep(period)
        except Exception:
            logger.exception("Unexpected error in LEDNetwork._poll()")
            raise

    def _packet_ready(self):
        data, (address, port) = self._socket.recvfrom(1500)
        if data[0] == FrameType.RESPONSE:
            bus_length, frame_count, rssi, unique_id, version = struct.unpack('>HIh32s32s', data[1:])
            unique_id = unique_id.rstrip(b'\x00').decode('UTF-8')
            version = version.rstrip(b'\x00').decode('UTF-8')
            if unique_id in self._devices:
                self._devices[unique_id].update(address, port, bus_length, frame_count, version, rssi)
            else:
                self._devices[unique_id] = self.Device(self, unique_id, self.format, address, port, bus_length, frame_count, version)
        else:
            logger.warning(f"Received unexpected frame {data} from {address}:{port}")

    def __len__(self):
        return len(self._devices)

    def __iter__(self):
        return iter(self._devices)

    def __getitem__(self, unique_id):
        return self._devices[unique_id]


class PixieRenderer:
    def __init__(self, **kwargs):
        self._net = None

    def purge(self):
        if self._net is not None:
            for device in self._net:
                device[:] = np.zeros(3)

    def destroy(self):
        if self._net is not None:
            self._net.close()
            self._net = None

    async def update(self, engine, node, **kwargs):
        if self._net is None:
            format = node.get('format', 1, str, 'GRB')
            address = node.get('address', 1, str, '')
            self._net = LEDNetwork(address, format=format)
        devices = {unique_id: self._net[unique_id] for unique_id in sorted(self._net)}
        render_list = []
        remaining_children = []
        for child in node.children:
            if child.kind == 'device':
                unique_id = node.get('id', 1, str)
                if unique_id in devices:
                    render_list.append((child, devices.pop(unique_id)))
                else:
                    remaining_children.append(child)
        while remaining_children and devices:
            unique_id, device = devices.popitem()
            child = remaining_children.pop(0)
            render_list.append((child, device))
        for node, device in render_list:
            offset = 0
            pixel_count = len(device)
            for child in node.children:
                if child.kind == 'segment':
                    color = np.array(child.get('color', 3, float, [0, 0, 0]))
                    length = child.get('length', 1, int)
                    if length is None:
                        length = pixel_count
                    elif length > 0:
                        length = min(length, pixel_count)
                    else:
                        continue
                    device[offset:offset+length] = color
                    offset += length
                    pixel_count -= length
                    if pixel_count == 0:
                        break


RENDERER_CLASS = PixieRenderer
