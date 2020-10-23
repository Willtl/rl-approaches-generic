import os, sys
import pygame
from enum import Enum
from pygame.locals import *


class Color(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    SHADOW = (192, 192, 192)
    LIGHTGREEN = (0, 255, 0)
    GREEN = (0, 200, 0)
    BLUE = (0, 0, 128)
    LIGHTBLUE = (0, 0, 255)
    RED = (200, 0, 0)
    LIGHTRED = (255, 100, 100)
    PURPLE = (102, 0, 102)
    LIGHTPURPLE = (153, 0, 153)


def load_image(name, colorkey=None):
    fullname = os.path.join('data', name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error as message:
        print('Cannot load image:', name)
        raise SystemExit(message)
    image = image.convert()
    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)
    return image, image.get_rect()


def load_sound(name):
    class NoneSound:
        def play(self): pass
    if not pygame.mixer:
        return NoneSound()
    fullname = os.path.join('data', name)
    try:
        sound = pygame.mixer.Sound(fullname)
    except pygame.error as message:
        print('Cannot load sound:', fullname)
        raise SystemExit(message)
    return sound

