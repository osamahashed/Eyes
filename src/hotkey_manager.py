import logging

from pynput import keyboard


class HotkeyManager:
    def __init__(self, config, action_queue):
        self.config = config
        self.action_queue = action_queue
        self.listener = None
        self.logger = logging.getLogger(__name__)

    def start(self):
        hotkeys = {
            self.config["hotkeys"]["rest_mode"]: lambda: self.enqueue("toggle_rest_mode"),
            self.config["hotkeys"]["toggle_modes"]: lambda: self.enqueue("cycle_click_mode"),
            self.config["hotkeys"]["start_calibration"]: lambda: self.enqueue("start_calibration"),
        }
        try:
            self.listener = keyboard.GlobalHotKeys(hotkeys)
            self.listener.start()
        except Exception as exc:
            self.logger.warning("Failed to start global hotkeys: %s", exc)

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def enqueue(self, action, payload=None):
        self.action_queue.put({"action": action, "payload": payload})
