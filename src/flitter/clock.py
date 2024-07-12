"""
Clock API

Provides a monotonically(*) increasing beat counter with a given tempo.

(*) The beat counter can actually go backwards by up to `backslip_limit` beat(s)
    when adjusting the phase with `set_phase`.
"""

import asyncio
import time

from numpy.polynomial.polynomial import polyfit


system_clock = time.perf_counter


class BeatCounter:
    def __init__(self, tempo=120, quantum=4, start=None):
        if start is None:
            start = system_clock()
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
        return self.beat_at_time(system_clock())

    @property
    def phase(self):
        return self.beat_at_time(system_clock()) % self._quantum

    def reset(self, tempo=120, quantum=4, start=None):
        if start is None:
            start = system_clock()
        self._start = start
        self._period = 60 / tempo
        self._quantum = quantum
        self._resync()

    def beat_at_time(self, timestamp):
        return (timestamp - self._start) / self._period

    def phase_at_time(self, timestamp):
        return (timestamp - self._start) / self._period % self._quantum

    def time_at_beat(self, beat):
        return beat * self._period + self._start

    def set_tempo(self, tempo, timestamp=None):
        if timestamp is None:
            timestamp = system_clock()
        period = 60 / tempo
        beat = (timestamp - self._start) / self._period
        self._start = timestamp - beat * period
        self._period = period
        self._resync()

    def set_quantum(self, quantum, timestamp=None):
        if timestamp is None:
            timestamp = system_clock()
        beat = (timestamp - self._start) / self._period
        phase = beat % self._quantum
        self._quantum = quantum
        self.set_phase(phase, timestamp)

    def set_phase(self, phase, timestamp=None, backslip_limit=0):
        if timestamp is None:
            timestamp = system_clock()
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
                now = system_clock()
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

    def __repr__(self):
        return f"BeatCounter(tempo={60/self._period:.1f}, quantum={self._quantum}, start={self._start:.1f})"


class TapTempo:
    def __init__(self, rounding=2):
        self._rounding = rounding
        self._taps = []

    def __len__(self):
        return len(self._taps)

    def tap(self, timestamp):
        self._taps.append(timestamp)

    def reset(self):
        self._taps = []

    def apply(self, counter, timestamp=None, backslip_limit=0):
        if timestamp is None:
            timestamp = system_clock()
        if self._taps:
            if len(self._taps) > 1:
                tzero, period = polyfit(range(len(self._taps)), self._taps, 1)
                tempo = 60 / period
                if self._rounding:
                    tempo = round(tempo * self._rounding) / self._rounding
                counter.set_tempo(tempo, timestamp)
                counter.set_phase(0, timestamp=tzero, backslip_limit=backslip_limit)
            else:
                counter.set_phase(0, timestamp=self._taps[0], backslip_limit=backslip_limit)
        self._taps = []
