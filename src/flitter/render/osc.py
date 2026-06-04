"""
Basic OSC protocol implementation
"""

import asyncio
import math
import socket
import struct
import time

from loguru import logger

from ..model import Vector


TIME = Vector.symbol('time')


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
            elif tag == 'T':
                arg = 0
            elif tag == 'F':
                arg = 1
            else:
                raise ValueError("Unrecognised type tag")
            args.append(arg)
        if len(data) != 0:
            raise ValueError("Junk at end of message")
        return cls(address, Vector.with_symbols(args))

    def __init__(self, address, args):
        self.address = address
        self.args = Vector.coerce(args)
        self._pattern = None

    def encode(self):
        tags = ','
        args_data = bytearray()
        for arg in self.args:
            if isinstance(arg, int):
                tags += 'i'
                args_data += encode_integer32(int(arg))
            elif isinstance(arg, float):
                if arg == 0:
                    tags += 'T'
                elif arg == 1:
                    tags += 'F'
                elif (symbol := Vector(arg).as_symbol()) is not None:
                    tags += 's'
                    args_data += encode_string(symbol)
                elif arg == math.floor(arg):
                    tags += 'i'
                    args_data += encode_integer32(int(arg))
                else:
                    tags += 'f'
                    args_data += encode_float32(arg)
            elif isinstance(arg, str):
                tags += 's'
                args_data += encode_string(arg)
            else:
                raise TypeError("Cannot encode argument: {!r}".format(arg))
        return encode_string(self.address) + encode_string(tags) + args_data

    def __bytes__(self):
        return self.encode()

    def __repr__(self):
        return f'OSCMessage({self.address!r}, {self.args!r})'


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
            key = next(iter(queue.keys()))
            blob = queue[key].encode()
            if blob is not None:
                blob = encode_blob(blob)
                if len(data) + len(blob) > max_size:
                    break
                data += blob
            queue.pop(key)
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
        self._cache = {}
        self._send_task = None
        self._queue = {}
        self._queue_not_empty = asyncio.Event()

    async def close(self):
        if self._send_task is not None:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
            self._send_task = None
        self._cache = {}
        self._queue = {}

    def enqueue(self, address, args, repeat):
        now = time.perf_counter()
        if address in self._cache:
            timestamp, last_args = self._cache[address]
            if args == last_args and (not repeat or now < timestamp + repeat):
                return
        message = OSCMessage(address, args)
        logger.trace("Enqueue message for {}:{} - {!r}", self._host, self._port, message)
        self._cache[address] = (now, args)
        self._queue[address] = message
        self._queue_not_empty.set()
        if self._send_task is None:
            self._send_task = asyncio.create_task(self._send_loop())

    async def _send_loop(self):
        logger.debug("Start OSC send loop for {}:{}", self._host, self._port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        if self._host == '<broadcast>':
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, True)
        try:
            sock.connect((self._host, self._port))
            while True:
                await self._queue_not_empty.wait()
                while self._queue:
                    try:
                        if len(self._queue) > 1:
                            bundle = OSCBundle.encode_from_queue(self._queue)
                            logger.trace("Send bundle to {}:{} - {!r}", self._host, self._port, bundle)
                            await asyncio.get_event_loop().sock_sendall(sock, bundle)
                        else:
                            _, message = self._queue.popitem()
                            message = message.encode()
                            logger.trace("Send message to {}:{} - {!r}", self._host, self._port, message)
                            await asyncio.get_event_loop().sock_sendall(sock, message)
                    except ConnectionRefusedError:
                        pass
                self._queue_not_empty.clear()
        except Exception:
            logger.exception("Error in OSC send loop for {}:{}", self._host, self._port)
        finally:
            sock.close()
            logger.debug("Stop OSC send loop for {}:{}", self._host, self._port)


class OSCReceiver:
    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._messages = {}
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def close(self):
        self._receive_task.cancel()
        try:
            await self._receive_task
        except asyncio.CancelledError:
            pass
        self._receive_task = None
        self._messages = {}

    def get_messages(self):
        messages = self._messages
        self._messages = {}
        return messages

    async def _receive_loop(self):
        logger.debug("Start OSC receive loop for {}:{}", self._host, self._port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        try:
            if self._host.startswith('224.'):
                sock.bind(('', self._port))
                membership = struct.pack("4sl", socket.inet_aton(self._host), socket.INADDR_ANY)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
            else:
                sock.bind((self._host, self._port))
            while True:
                data = await asyncio.get_event_loop().sock_recv(sock, 1400)
                now = time.perf_counter()
                packet = decode_oscpacket(data)
                if isinstance(packet, OSCBundle):
                    logger.trace("Receive bundle on {}:{} - {!r}", self._host, self._port, data)
                    for message in packet.elements:
                        self._messages[message.address] = now, message.args
                else:
                    logger.trace("Receive message on {}:{} - {!r}", self._host, self._port, data)
                    self._messages[packet.address] = now, packet.args
        except Exception:
            logger.exception("Error in OSC receive loop for {}:{}", self._host, self._port)
        finally:
            sock.close()
            logger.debug("Stop OSC receive loop for {}:{}", self._host, self._port)


class OSC:
    def __init__(self, **kwargs):
        self._senders = {}
        self._receivers = {}

    async def purge(self):
        pass

    async def destroy(self):
        for sender in self._senders.values():
            await sender.close()
        self._senders = {}
        for receiver in self._receivers.values():
            await receiver.close()
        self._receivers = {}

    async def update(self, engine, node, **kwargs):
        senders = {}
        receivers = {}
        for child in node.children:
            if child.kind == 'send':
                host = child.get('host', 1, str, 'localhost')
                port = child.get('port', 1, int, 8000)
                sender = self._senders.pop((host, port), None)
                if sender is None:
                    sender = OSCSender(host, port)
                senders[(host, port)] = sender
                self.collect_sender(sender, child)
            elif child.kind == 'receive':
                host = child.get('host', 1, str, 'localhost')
                port = child.get('port', 1, int, 8000)
                receiver = self._receivers.pop((host, port), None)
                if receiver is None:
                    receiver = OSCReceiver(host, port)
                receivers[(host, port)] = receiver
                self.collect_receiver(engine, receiver, child)
        for sender in self._senders.values():
            await sender.close()
        self._senders = senders
        for receiver in self._receivers.values():
            await receiver.close()
        self._receivers = receivers

    def collect_sender(self, sender, node):
        for child in node.children:
            if child.kind == 'method' and 'address' in child and 'arguments' in child:
                address = child.get('address', 1, str)
                args = child['arguments']
                repeat = child.get('repeat', 1, int)
                sender.enqueue(address, args, repeat)

    def collect_receiver(self, engine, receiver, node):
        messages = receiver.get_messages()
        for child in node.children:
            if child.kind == 'method' and 'address' in child and 'state' in child:
                address = child.get('address', 1, str)
                state_key = child['state']
                if address in messages and state_key:
                    timestamp, args = messages[address]
                    engine.state[state_key] = args
                    engine.state[state_key.concat(TIME)] = timestamp
