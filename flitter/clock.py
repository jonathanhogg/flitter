"""
Clock API

Provides a monotonically(*) increasing beat counter with a given tempo.

(*) The beat counter can actually go backwards by up to `backslip_limit` beat(s)
    when adjusting the phase with `set_phase`.
"""

import asyncio
import time


class BeatCounter:
    @staticmethod
    def clock():
        return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)

    def __init__(self, tempo=120, quantum=4, start=None):
        if start is None:
            start = self.clock()
        self._start = start
        self._period = 60 / tempo
        self._quantum = quantum
        self._resync_events = set()

    @property
    def tempo(self):
        return 60 / self._period

    @tempo.setter
    def tempo(self, tempo):
        self.set_tempo(tempo)

    @property
    def quantum(self):
        return self._quantum

    @quantum.setter
    def quantum(self, quantum):
        self.set_quantum(quantum)

    @property
    def start(self):
        return self._start

    @property
    def beat(self):
        return self.beat_at_time(self.clock())

    @property
    def phase(self):
        return self.beat_at_time(self.clock()) % self._quantum

    def beat_at_time(self, timestamp):
        return (timestamp - self._start) / self._period

    def time_at_beat(self, beat):
        return beat * self._period + self._start

    def set_tempo(self, tempo, timestamp=None):
        if timestamp is None:
            timestamp = self.clock()
        period = 60 / tempo
        beat = (timestamp - self._start) / self._period
        self._start = timestamp - beat * period
        self._period = period
        self._resync()

    def set_quantum(self, quantum, timestamp=None):
        if timestamp is None:
            timestamp = self.clock()
        beat = (timestamp - self._start) / self._period
        phase = beat % self._quantum
        self._quantum = quantum
        self.set_phase(phase, timestamp)

    def set_phase(self, phase, timestamp=None, backslip_limit=0):
        if timestamp is None:
            timestamp = self.clock()
        beat = (timestamp - self._start) / self._period
        adjustment = (phase - beat) % self._quantum
        if adjustment > self._quantum - backslip_limit:
            adjustment -= self._quantum
        self._start -= adjustment * self._period
        self._resync()

    def _resync(self):
        for resync_event in self._resync_events:
            resync_event.set()

    async def wait_for_beat(self, beat):
        resync_event = asyncio.Event()
        self._resync_events.add(resync_event)
        try:
            while True:
                timestamp = self.time_at_beat(beat)
                now = self.clock()
                if now >= timestamp:
                    return
                try:
                    await asyncio.wait_for(resync_event.wait(), timeout=timestamp - now)
                except asyncio.TimeoutError:
                    pass
                else:
                    resync_event.clear()
        finally:
            self._resync_events.remove(resync_event)
