"""
A simple replacement for pygame.time.set_timer, which does not exist in
the browser version of pygame.

set_timer_once(event, ms) remembers "send this event in ms milliseconds".
fire_due() is called once per frame and sends any events whose time is up,
exactly as pygame's own timer would. Works the same on desktop and browser.
"""
import time
import pygame

_pending = []   # list of (time_due, event_type)


def set_timer_once(event_type, millis):
    _pending.append((time.time() + millis / 1000.0, event_type))


def fire_due():
    now = time.time()
    due = [event_type for t, event_type in _pending if t <= now]
    _pending[:] = [(t, e) for t, e in _pending if t > now]
    for event_type in due:
        pygame.event.post(pygame.event.Event(event_type))
