"""
Basic OSC protocol implementation
"""

import asyncio
import socket
import struct
import time

from loguru import logger

from .. import name_patch


logger = name_patch(logger, __name__)


def decode_string(data):
    i = 4
    n = len(data)
    while i < n and data[i-1] != 0:
        i += 4
    remains = data[i:]
    while i and data[i-1] == 0:
        i -= 1
    return bytes(data[:i]).decode('utf8'), remains


def encode_string(s):
    data = s.encode('utf8') + b'\0'
    if len(data) % 4:
        data += b'\0' * (4 - len(data) % 4)
    return data


def decode_integer32(data):
    i = struct.unpack_from('>i', data)[0]
    return i, data[4:]


def encode_integer32(i):
    return struct.pack('>i', i)


def decode_float32(data):
    f = struct.unpack_from('>f', data)[0]
    return f, data[4:]


def encode_float32(f):
    return struct.pack('>f', f)


def decode_blob(data):
    n = struct.unpack_from('>i', data)[0]
    blob = data[4:n + 4]
    if n % 4:
        n += 4 - n % 4
    return blob, data[4 + n:]


def encode_blob(data):
    data = struct.pack('>i', len(data)) + data
    while len(data) % 4:
        data += b'\0'
    return data


def decode_timetag(data):
    timetag = NetworkTime(*struct.unpack_from('>II', data))
    return timetag, data[8:]


def encode_timetag(timetag):
    return struct.pack('>II', timetag.seconds, timetag.fraction)


def decode_oscpacket(data):
    data = memoryview(data)
    if data[:8] == b'#bundle\0':
        return OSCBundle.decode(data)
    return OSCMessage.decode(data)


class NetworkTime:
    EpochTimeOffset = 2208988800
    Scale = 1 << 32
    Milliseconds = Scale / 1000

    @classmethod
    def from_time(cls, t=None):
        return cls(0, int(((t if t is not None else time.time()) + cls.EpochTimeOffset) * cls.Scale))

    def __init__(self, seconds, fraction):
        self.seconds = seconds + fraction // self.Scale
        self.fraction = fraction % self.Scale

    def to_time(self):
        return self.seconds + self.fraction / self.Scale - self.EpochTimeOffset

    def isoformat(self):
        tm = time.gmtime(self.seconds - self.EpochTimeOffset)
        microseconds = min(int(self.fraction / self.Milliseconds), 999999)
        return "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}.{:06d}Z".format(*tm[:6] + (microseconds,))

    def add_microseconds(self, microseconds):
        return NetworkTime(self.seconds, self.fraction + int(microseconds * self.Milliseconds))

    def add_seconds(self, seconds):
        return NetworkTime(self.seconds, self.fraction + int(seconds * self.Scale))

    def __eq__(self, other):
        return (self.seconds, self.fraction) == (other.seconds, other.fraction)

    def __lt__(self, other):
        return (self.seconds, self.fraction) < (other.seconds, other.fraction)

    def __gt__(self, other):
        return (self.seconds, self.fraction) > (other.seconds, other.fraction)

    def __le__(self, other):
        return (self.seconds, self.fraction) <= (other.seconds, other.fraction)

    def __ge__(self, other):
        return (self.seconds, self.fraction) >= (other.seconds, other.fraction)

    def __sub__(self, other):
        return self.seconds - other.seconds + (self.fraction - other.fraction) / self.Scale

    def __repr__(self):
        return "NetworkTime({!r}, {!r})".format(self.seconds, self.fraction)


class OSCMessage:
    @classmethod
    def decode(cls, data):
        if len(data) % 4:
            raise ValueError("Bad message size")
        address, data = decode_string(data)
        if not address or not address[0] == '/':
            raise ValueError("Bad address")
        tags, data = decode_string(data)
        if not tags.startswith(','):
            raise ValueError("Bad type tag string")
        args = []
        for tag in tags[1:]:
            if tag == 'i':
                arg, data = decode_integer32(data)
            elif tag == 'f':
                arg, data = decode_float32(data)
            elif tag == 's':
                arg, data = decode_string(data)
            elif tag == 'b':
                arg, data = decode_blob(data)
            elif tag == 't':
                arg, data = decode_timetag(data)
            elif tag == 'T':
                arg = True
            elif tag == 'F':
                arg = False
            elif tag == 'N':
                arg = None
            else:
                raise ValueError("Unrecognised type tag")
            args.append(arg)
        if len(data) != 0:
            raise ValueError("Junk at end of message")
        return cls(address, *args)

    def __init__(self, address, *args):
        self.address = address
        self.args = args
        self._pattern = None

    def encode(self):
        tags = ','
        args_data = b''
        for arg in self.args:
            if arg is None:
                tags += 'N'
            elif isinstance(arg, bool):
                tags += 'T' if arg else 'F'
            elif isinstance(arg, int):
                tags += 'i'
                args_data += encode_integer32(arg)
            elif isinstance(arg, float):
                tags += 'f'
                args_data += encode_float32(arg)
            elif isinstance(arg, str):
                tags += 's'
                args_data += encode_string(arg)
            elif isinstance(arg, bytes):
                tags += 'b'
                args_data += encode_blob(arg)
            elif isinstance(arg, NetworkTime):
                tags += 't'
                args_data += encode_timetag(arg)
            else:
                raise TypeError("Cannot encode argument: {!r}".format(arg))
        return encode_string(self.address) + encode_string(tags) + args_data

    def __bytes__(self):
        return self.encode()

    def __repr__(self):
        return "OSCMessage({!r}{})".format(self.address, ''.join(', {!r}'.format(arg) for arg in self.args))


class OSCBundle:
    @classmethod
    def decode(cls, data):
        if len(data) % 4:
            raise ValueError("Bad message size")
        identifier, data = decode_string(data)
        if identifier != '#bundle':
            raise ValueError("Bad identifier")
        timetag, data = decode_timetag(data)
        elements = []
        while data:
            contents, data = decode_blob(data)
            elements.append(decode_oscpacket(contents))
        return cls(timetag, elements)

    @classmethod
    def encode_from_queue(cls, queue, timetag=None, max_size=1400):
        data = encode_string('#bundle')
        data += encode_timetag(timetag if timetag is not None else NetworkTime.from_time())
        while queue:
            blob = queue[0].encode()
            if blob is not None:
                blob = encode_blob(blob)
                if len(data) + len(blob) > max_size:
                    break
                data += blob
            queue.pop(0)
        return data

    def __init__(self, timetag, elements):
        self.timetag = timetag
        self.elements = list(elements)

    def encode(self):
        data = encode_string('#bundle')
        data += encode_timetag(self.timetag)
        for element in self.elements:
            data += encode_blob(element.encode())
        return data

    def __bytes__(self):
        return self.encode()

    def __repr__(self):
        return "OSCBundle({!r}, {!r})".format(self.timetag, self.elements)


class OSCSender:
    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._socket = None

    async def _send(self, data):
        logger.trace("Send: {!r}", data)
        if self._socket is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            await asyncio.get_event_loop().sock_connect(sock, (self._host, self._port))
            self._socket = sock
        try:
            await asyncio.get_event_loop().sock_sendall(self._socket, data)
        except ConnectionRefusedError:
            pass

    async def send_message(self, address, *args):
        await self._send(OSCMessage(address, *args).encode())

    async def send_bundle_from_queue(self, *args, **kwargs):
        await self._send(OSCBundle.encode_from_queue(*args, **kwargs))


class OSCReceiver:
    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._socket = None

    async def receive(self, mtu=1500):
        if self._socket is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            sock.bind((self._host, self._port))
            self._socket = sock
        data = await asyncio.get_event_loop().sock_recv(self._socket, mtu)
        logger.trace("Receive: {!r}", data)
        return decode_oscpacket(data)
