"""
Nick Cantalupa and Sean Duffy
Computer Vision Spring 2025

This file contains utility functions for the object detection application.
It includes functions for loading adversarial patches, getting user input, and creating patch overlay functions.
"""

import sys
import os
from typing import List

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton


class InputDialog(QDialog):
    """
    A dialog window for obtaining object label inputs from the user
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Input Dialog")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        self.label_id = QLabel("Object ID:")
        layout.addWidget(self.label_id)

        self.input_id = QLineEdit()
        layout.addWidget(self.input_id)

        self.label_name = QLabel("Name:")
        layout.addWidget(self.label_name)

        self.input_name = QLineEdit()
        layout.addWidget(self.input_name)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.accept)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)

    def get_inputs(self):
        """
        Retrieve user input values for ID and name.

        :return: A tuple containing the text from the input_id and input_name input widgets
        """
        return self.input_id.text(), self.input_name.text()


def get_name() -> tuple[int, str] | tuple[None, None]:
    """
    Retrieves a unique identifier and a name from user input.

    :return: A tuple containing an ID and a label. (int, str) if no invalid inputs else (None, None)
    """
    app = QApplication(sys.argv)
    dialog = InputDialog()
    if dialog.exec_() == QDialog.Accepted:
        object_id, name = dialog.get_inputs()
    else:
        print('Invalid input(s)')
        return None, None
    app.quit()

    return int(object_id), name.lower().strip()


def load_adversarial_patches() -> List[np.ndarray]:
    """
    Loads adversarial patches from a specified directory.
    Stored in a list as cv2 image matrices.

    :param patch_dir: Path to the directory containing adversarial image patches. Defaults
                      to 'data/adversarial'.
    :return: A list of patches as cv2 image matrices
    """
    patches = []
    patch_dir = 'data/adversarial'
    if os.path.exists(patch_dir):
        for filename in os.listdir(patch_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(patch_dir, filename)
                patch = cv2.imread(img_path)
                if patch is not None:
                    patches.append(patch)
    return patches


def get_patch_func(state_dict: dict, patches: list) -> callable:
    """
    Generates a function that alters the state dictionary based on mouse events.

    :param state_dict: Dictionary to hold the state information relevant to the patch
    :param patches: List of patches as cv2 image matrices

    :return: A function that handles mouse events and modifies the state_dict
    """
    patch_count = len(patches)

    # Function with signature matching cv2.setMouseCallback
    def patch(event, x, y, flags, param):
        # Update coordinates as mouse moves
        if event == cv2.EVENT_MOUSEMOVE:
            state_dict['patch_coords'] = (x, y)
        # Cycle through patches on left click
        elif event == cv2.EVENT_LBUTTONDOWN:
            if state_dict['patch'] is None:
                state_dict['patch'] = 0
            elif not (state_dict['patch'] is None) and state_dict['patch'] + 1 < patch_count:
                state_dict['patch'] = state_dict['patch'] + 1
                state_dict['patch_scale'] = 1.0
            else:
                state_dict['patch'] = None
        # Remove patch on right click
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state_dict['patch']:
                state_dict['patch'] = None

    return patch
