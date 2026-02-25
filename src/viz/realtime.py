"""PyBullet real-time debug visualization helpers."""

import numpy as np
import pybullet as p


class RealtimeViz:
    """Draw bearing lines, labels, and comm graph edges in PyBullet GUI."""

    def __init__(self, client_id: int):
        self.client = client_id
        self._items = []

    def clear(self):
        for item_id in self._items:
            p.removeUserDebugItem(item_id, physicsClientId=self.client)
        self._items = []

    def draw_text(self, text: str, position: np.ndarray, color: list = None):
        color = color or [1, 1, 1]
        item = p.addUserDebugText(
            text, position.tolist(),
            textColorRGB=color, textSize=1.2,
            lifeTime=0, physicsClientId=self.client,
        )
        self._items.append(item)

    def draw_drone_labels(self, positions: np.ndarray, labels: list[str]):
        for pos, label in zip(positions, labels):
            offset = pos + np.array([0, 0, 0.15])
            self.draw_text(label, offset)

    def draw_target_label(self, position: np.ndarray, label: str = "TARGET"):
        offset = position + np.array([0, 0, 0.15])
        self.draw_text(label, offset, color=[1, 0.2, 0.2])
