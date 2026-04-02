from pynput.mouse import Controller, Button
import time

class MouseController:
    def __init__(self):
        self.mouse = Controller()

    def move_cursor(self, x, y):
        self.mouse.position = (int(x), int(y))

    def click(self, button=Button.left):
        self.mouse.click(button)

    def scroll(self, dx, dy):
        self.mouse.scroll(dx, dy)
