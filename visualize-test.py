#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numbers import Real
from typing import Callable, Iterable

import math
import numpy as np
import pyaudio
# import scipy.ndimage
# import scipy.signal
import time

from collections import deque
from ctypes import byref, c_int

try:
    import sdl2
except ImportError:
    sdl2 = None
    sdlgfx = None
else:
    try:
        import sdl2.sdlgfx as sdlgfx
    except ImportError:
        sdlgfx = None

try:
    import curses
except ImportError:
    curses = None


DEFAULT_SAMPLING_FREQUENCY = 48000


def generate_sine(frequency, duration: Real = 1,
                  sampling_frequency: Real = DEFAULT_SAMPLING_FREQUENCY,
                  endpoint=False):

    n_real = duration * sampling_frequency
    n_int = int(math.ceil(n_real))

    times = np.linspace(0, duration * (n_int / n_real), num=n_int + 1,
                        dtype=np.float_)
    if not endpoint:
        times = times[:-1]

    return np.sin(np.pi * 2 * frequency * times)


class ShortTimeFourierTransform:
    ANALYZE_START, ANALYZE_CENTER, ANALYZE_END = range(3)

    def __init__(self, duration: Real = 1,
                 sampling_frequency: Real = DEFAULT_SAMPLING_FREQUENCY):

        self.half_size = max(int(math.ceil(duration * sampling_frequency / 2)),
                             1)
        self.window_size = self.half_size * 2
        self.min_size = self.window_size * 2

        self.sampling_frequency = sampling_frequency
        self.nyquist_frequency = sampling_frequency / 2
        self.rayleigh_frequency = sampling_frequency / self.window_size

        self._p = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi,
                                                 num=self.window_size,
                                                 endpoint=False))
        self._q = 1 - self._p

    def analyze(self, amplitudes: np.ndarray, where=ANALYZE_END) -> np.ndarray:

        if len(amplitudes) < self.min_size:
            return np.empty((0,))

        if where == self.ANALYZE_END:
            centre = len(amplitudes) - self.window_size
        elif where == self.ANALYZE_START:
            centre = self.window_size
        elif where == self.ANALYZE_CENTER:
            centre = len(amplitudes) // 2
        else:
            raise ValueError("unknown \"where\" value")

        t1, t2, t3, t4, t5 = (centre - self.window_size,
                              centre - self.half_size,
                              centre,
                              centre + self.half_size,
                              centre + self.window_size)

        complement = (self._p * amplitudes[t1:t3] +
                      self._q * amplitudes[t3:t5])

        window = np.concatenate([
            self._p[::2] * amplitudes[t2:t3] +
            self._q[::2] * complement[self.half_size:],
            self._q[::2] * amplitudes[t3:t4] +
            self._p[::2] * complement[:self.half_size]
        ])

        frequency_powers = np.abs(np.fft.rfft(window) / self.window_size) ** 2
        frequency_powers[1:self.half_size] *= 2
        return frequency_powers


def aggregate(values: Iterable, n: int,
              dtype: np.dtype=None,
              func: Callable=np.mean) -> Iterable:

    agg = None if dtype is None else np.empty((n,), dtype=dtype)
    i = 0

    for value in values:
        if agg is None:
            agg = np.empty((n,), dtype=type(value))

        agg[i] = value
        i += 1

        if i == n:
            yield func(agg)
            i = 0

    if i:
        agg[i:].fill(0.0)
        yield func(agg)


class MyApp:
    def __init__(self, min_level=-100.0, max_level=0.0, frame_rate=60):

        self.min_level = min_level
        self.max_level = max_level

        self._event = None
        self._rect = None

        self.sdl2_initialized = False
        self.window = None
        self.renderer = None
        self.width = 80
        self.height = 24
        self.mouse_x = None
        self.mouse_y = None
        self.pyaudio = None
        self.stream = None
        self.stdscr = None
        self._color_cache = None
        self.fps_manager = None

        self.frame_rate = None
        self.duty_cycle = None
        self._target_delay = None
        self._last_time = None

        if sdl2 and sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) == 0:
            self._event = sdl2.SDL_Event()
            self._rect = sdl2.SDL_Rect()
            self.sdl2_initialized = True

        if self.sdl2_initialized:
            self.window = sdl2.SDL_CreateWindow(b"Foobar",
                                                sdl2.SDL_WINDOWPOS_CENTERED,
                                                sdl2.SDL_WINDOWPOS_CENTERED,
                                                960, 720,
                                                sdl2.SDL_WINDOW_SHOWN |
                                                sdl2.SDL_WINDOW_RESIZABLE)

        if self.window:
            self.renderer = sdl2.SDL_CreateRenderer(
                    self.window, -1,
                    sdl2.SDL_RENDERER_ACCELERATED |
                    sdl2.SDL_RENDERER_PRESENTVSYNC
            )

            if not self.renderer:
                raise RuntimeError("could not create renderer")

            sdl2.SDL_SetRenderDrawBlendMode(self.renderer,
                                            sdl2.SDL_BLENDMODE_NONE)

        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(rate=DEFAULT_SAMPLING_FREQUENCY,
                                        channels=1,
                                        format=pyaudio.paFloat32,
                                        input=True)

        if not self.renderer and curses:
            self.stdscr = curses.initscr()
            self.init_curses_colors()
            curses.noecho()
            curses.cbreak()
            curses.mousemask(curses.BUTTON1_PRESSED | curses.BUTTON1_RELEASED |
                             curses.BUTTON2_PRESSED | curses.BUTTON2_RELEASED |
                             curses.BUTTON3_PRESSED | curses.BUTTON3_RELEASED)
            curses.mouseinterval(0)
            self.stdscr.keypad(True)
            self.stdscr.nodelay(True)

        if not self.renderer and not self.stdscr:
            raise RuntimeError("no usable display found")

        if sdlgfx:
            self.fps_manager = sdlgfx.FPSManager()
            sdlgfx.SDL_initFramerate(self.fps_manager)

        self.set_frame_rate(frame_rate)

        self.update_render_size()

    def clean_up_curses(self):

        if self.stdscr:
            self.stdscr.keypad(False)
            curses.mousemask(0)
            curses.nocbreak()
            curses.echo()
            curses.endwin()
            self.stdscr = None

    def __del__(self):

        self.clean_up_curses()

        if self.stream:
            self.stream.close()
            self.stream = None

        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None

        if self.renderer:
            sdl2.SDL_DestroyRenderer(self.renderer)
            self.renderer = None

        if self.window:
            sdl2.SDL_DestroyWindow(self.window)
            self.window = None

        if self.sdl2_initialized:
            sdl2.SDL_Quit()
            self.sdl2_initialized = False

    def init_curses_colors(self):
        if not curses.has_colors():
            return

        curses.start_color()
        self._color_cache = {}
        # noinspection PyUnresolvedReferences
        num_colors = min(curses.COLORS, curses.COLOR_PAIRS)

        pair0 = curses.pair_content(0)
        for i in range(1, num_colors):
            pair = (i, 0)
            if pair == pair0:
                pair = (0, 0)

            curses.init_pair(i, *pair)

        if not curses.can_change_color() or num_colors <= 8:
            return

        i = 8
        tmp = num_colors - i
        num_r = num_g = num_b = int(math.ceil(tmp ** (1 / 3)))
        if num_r * num_g * num_b > tmp:
            num_g -= 1
        if num_r * num_g * num_b > tmp:
            num_r -= 1
        if num_r * num_g * num_b > tmp:
            num_b -= 1

        if num_r > 1 and num_g > 1 and num_b > 1:
            max_r, max_g, max_b = num_r - 1, num_g - 1, num_b - 1
            half_r, half_g, half_b = max_r // 2, max_g // 2, max_b // 2

            for ir in range(num_r):
                r = (ir * 1000 + half_r) // max_r

                for ig in range(num_g):
                    g = (ig * 1000 + half_g) // max_g

                    for ib in range(num_b):
                        b = (ib * 1000 + half_b) // max_b

                        curses.init_color(i, r, g, b)
                        i += 1

        if i >= num_colors:
            return

        max_gray = num_colors - i + 1
        half_gray = max_gray // 2

        for igray in range(1, max_gray):
            gray = (igray * 1000 + half_gray) // max_gray

            curses.init_color(i, gray, gray, gray)
            i += 1

    def find_color_pair(self, r, g, b, a=255):
        if not curses.has_colors():
            return 0

        if a != 255:
            r = (r * a + 127) // 255
            g = (g * a + 127) // 255
            b = (b * a + 127) // 255

        i = self._color_cache.get((r, g, b))
        if i is not None:
            return i

        best = 0
        best_dist = self._curses_color_dist(0, r, g, b)
        # noinspection PyUnresolvedReferences
        for i in range(1, curses.COLOR_PAIRS):
            dist = self._curses_color_dist(i, r, g, b)
            if dist < best_dist:
                best_dist = dist
                best = i

        self._color_cache[(r, g, b)] = best
        return best

    @staticmethod
    def _curses_color_dist(i, r, g, b):
        ir, ig, ib = curses.color_content(curses.pair_content(i)[0])
        return ((r * 1000 - ir * 255) ** 2 +
                (g * 1000 - ig * 255) ** 2 +
                (b * 1000 - ib * 255) ** 2)

    def update_render_size(self):

        if self.renderer:
            width = c_int()
            height = c_int()
            sdl2.SDL_GetRendererOutputSize(self.renderer,
                                           byref(width),
                                           byref(height))

            self.width = width.value
            self.height = height.value

        elif self.stdscr:
            self.height, self.width = self.stdscr.getmaxyx()

    def process_events(self) -> bool:

        if self.window:
            while sdl2.SDL_PollEvent(byref(self._event)):
                if self._event.type == sdl2.SDL_QUIT:
                    return False

                elif self._event.type == sdl2.SDL_WINDOWEVENT:
                    window_event = self._event.window
                    if window_event.event == sdl2.SDL_WINDOWEVENT_RESIZED:
                        self.width = window_event.data1
                        self.height = window_event.data2

                    elif window_event.event == sdl2.SDL_WINDOWEVENT_LEAVE:
                        self.mouse_x = None
                        self.mouse_y = None

                elif self._event.type == sdl2.SDL_MOUSEMOTION:
                    self.mouse_x = self._event.motion.x
                    self.mouse_y = self._event.motion.y

                elif self._event.type == sdl2.SDL_KEYDOWN:
                    keysym = self._event.key.keysym
                    if keysym.sym in [sdl2.SDLK_ESCAPE, sdl2.SDLK_q,
                                      sdl2.SDLK_x]:
                        return False

            return True

        elif self.stdscr:
            while True:
                try:
                    wch = self.stdscr.get_wch()
                except curses.error:
                    break

                if wch == curses.KEY_RESIZE:
                    curses.update_lines_cols()
                    self.height, self.width = self.stdscr.getmaxyx()

                elif wch == curses.KEY_MOUSE:
                    try:
                        mouse = curses.getmouse()
                    except curses.error:
                        pass
                    else:
                        if mouse[4] & curses.BUTTON1_PRESSED:
                            self.mouse_x, self.mouse_y = mouse[1:3]
                        elif mouse[4] & (curses.BUTTON2_PRESSED |
                                         curses.BUTTON3_PRESSED):
                            self.mouse_x = self.mouse_y = None

                elif wch in ['\x1b', 'q', 'x']:
                    return False

            return True

        return False

    def render_clear(self):

        if self.renderer:
            sdl2.SDL_SetRenderDrawColor(self.renderer, 0, 0, 0, 255)
            sdl2.SDL_RenderClear(self.renderer)

        elif self.stdscr:
            self.stdscr.erase()

    def render_rect(self, x, y, w, h, r=255, g=255, b=255, a=255):

        if self.renderer:
            sdl2.SDL_SetRenderDrawColor(self.renderer, r, g, b, a)
            self._rect.x, self._rect.y, self._rect.w, self._rect.h = x, y, w, h
            sdl2.SDL_RenderFillRect(self.renderer, self._rect)

        elif self.stdscr:
            if h > 0 and y < self.height and y + h > 0:
                pair = self.find_color_pair(r, g, b, a)
                self.stdscr.attrset(curses.A_REVERSE | curses.color_pair(pair))
                for col in range(max(x, 0), min(x + w, self.width)):
                    self.stdscr.vline(y, col, b'+', h)
                self.stdscr.attrset(curses.A_NORMAL | curses.color_pair(0))

    def render_text(self, text, line, r=255, g=255, b=255, a=255):

        if self.renderer and sdlgfx:
            sdlgfx.stringRGBA(self.renderer, 1, 1 + 9 * line,
                              str(text).encode('cp437', 'replace'),
                              r, g, b, a)

        elif self.stdscr:
            pair = self.find_color_pair(r, g, b, a)
            self.stdscr.addstr(line, 0, str(text), curses.color_pair(pair))

    def render_present(self):

        if self.renderer:
            sdl2.SDL_RenderPresent(self.renderer)

        elif self.stdscr:
            self.stdscr.refresh()

    def set_frame_rate(self, frame_rate):

        if self.fps_manager:
            sdlgfx.SDL_setFramerate(self.fps_manager, math.ceil(frame_rate))

        elif self.sdl2_initialized:
            self._target_delay = math.floor(1000 / frame_rate)
            self._last_time = sdl2.SDL_GetTicks()

        else:
            self._target_delay = 1 / frame_rate
            self._last_time = time.monotonic()

    def render_delay(self):

        if self.fps_manager:
            delta_ticks = sdlgfx.SDL_framerateDelay(self.fps_manager)
            self.frame_rate = 1000 / delta_ticks if delta_ticks > 0 else None

        elif self.sdl2_initialized:
            while True:
                current_ticks = sdl2.SDL_GetTicks()
                delta_ticks = (current_ticks - self._last_time) & 0xffffffff

                if delta_ticks >= self._target_delay:
                    self.frame_rate = 1000 / delta_ticks
                    break

                sdl2.SDL_Delay(self._target_delay - delta_ticks)

            self._last_time = current_ticks

        else:
            while True:
                current_time = time.monotonic()
                delta_time = current_time - self._last_time

                if delta_time >= self._target_delay:
                    self.frame_rate = 1 / delta_time
                    break

                if delta_time < 0:
                    self.frame_rate = None
                    break

                time.sleep(self._target_delay - delta_time)

            self._last_time = current_time

    def loop(self):

        queue = deque()
        queue.append(np.empty((0,)))
        fill = 0
        first = 0

        # import itertools
        # progression = itertools.cycle(np.linspace(100, 130, num=1800,
        #                                           endpoint=False))

        stft = ShortTimeFourierTransform(duration=0.2)

        while self.process_events():
            self.render_delay()

            available = self.stream.get_read_available()
            if available:
                while True:
                    chunk = np.frombuffer(self.stream.read(available),
                                          dtype=np.float32)

                    queue.append(chunk)
                    fill += len(chunk)

                    available = self.stream.get_read_available()
                    if not available:
                        break

                if first == 0:
                    first = len(queue[0])

                while fill - first >= stft.min_size:
                    queue.popleft()
                    fill -= first
                    first = len(queue[0])

            frequency_powers = stft.analyze(np.concatenate(queue))
            if not len(frequency_powers):
                continue

            # frequency = next(progression)
            # amplitudes = generate_sine(frequency, duration=3)
            # frequency_powers = stft.analyze(amplitudes)

            energy = math.fsum(frequency_powers)

            with np.errstate(divide='ignore'):
                levels = 10 * np.log10(frequency_powers)

            # levels = scipy.signal.savgol_filter(levels, 65, 3)

            # levels = scipy.ndimage.filters.gaussian_filter1d(
            #         levels, 20.0, mode='nearest', truncate=100.0
            # )

            # levels = scipy.signal.wiener(levels, 33)

            self.render_clear()

            self._handle_horizontal(levels[1:961])

            # w = 1
            # for i, level in enumerate(levels[1:]):
            #     x = i * w
            #     if x >= self.width:
            #         break
            #
            #     if level <= self.min_level:
            #         continue
            #
            #     h = round((float(level) - self.min_level) /
            #               (self.max_level - self.min_level) * self.height)
            #     if h <= 0:
            #         continue
            #
            #     y = self.height - h
            #     self.render_rect(x, y, w, h, 255, 255, 0)

            if self.frame_rate:
                self.render_text(f"{self.frame_rate:.1f} FPS", 0,
                                 0, 255, 255, 192)

            if self.duty_cycle is not None:
                self.render_text(f"{self.duty_cycle*100.0:.1f}%", 1,
                                 0, 255, 255, 192)

            self.render_text(f"{self.width} x {self.height}", 2,
                             255, 0, 0, 192)
            self.render_text(f"{fill}", 3, 0, 255, 0, 192)
            self.render_text(f"{energy:.4f}", 4, 255, 255, 0, 192)

            if self.mouse_x is not None:
                i = int((self.mouse_x + 0.5) / self.width * len(levels[1:961])
                        + 1)
                self.render_text(f"{i * stft.rayleigh_frequency:.2f} Hz", 6)

            if self.mouse_y is not None:
                y = (self.height - self.mouse_y - 0.5) / self.height
                y = y * (self.max_level - self.min_level) + self.min_level
                self.render_text(f"{y:.1f} dB", 7)

            self.render_present()

    def _handle_horizontal(self, levels):
        if not len(levels):
            return

        xstep = self.width / len(levels)
        xfrac = 0
        x = 0
        combine = 0
        for level in levels:
            xrem = xstep

            if xfrac != 0:
                xsub = 1 - xfrac
                xrem -= xsub

                if xrem < 0:
                    combine += level * xstep
                    xfrac += xstep
                    continue

                combine += level * xsub
                h = round((float(combine) - self.min_level) /
                          (self.max_level - self.min_level) * self.height)
                if h > 0:
                    self.render_rect(x, self.height - h, 1, h, 255, 255, 0)
                xfrac = 0
                x += 1

            if xrem >= 1:
                w = int(xrem)
                h = round((float(level) - self.min_level) /
                          (self.max_level - self.min_level) * self.height)
                if h > 0:
                    self.render_rect(x, self.height - h, w, h, 255, 255, 0)
                xrem -= w
                x += w

            if xrem <= 0:
                continue

            combine = level * xrem
            xfrac = xrem

        if xfrac != 0:
            h = round((float(combine) - self.min_level) /
                      (self.max_level - self.min_level) * self.height)
            if h > 0:
                self.render_rect(x, self.height - h, 1, h, 255, 255, 0)


def main():

    app = MyApp()

    try:
        app.loop()
    except Exception:
        app.clean_up_curses()
        raise


if __name__ == '__main__':
    import sys
    sys.exit(main())
