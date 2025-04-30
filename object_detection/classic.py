"""
Nick Cantalupa and Sean Duffy
Computer Vision Spring 2025

This file contains the classic object detection functionalities including thresholding, morphological filtering,
connected components, feature calculation and database management.
"""

import math
from typing import Optional

import cv2
import numpy as np
import random
import json
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
MIN_COMPONENT_AREA = config['MIN_COMPONENT_AREA']
MAX_COMPONENT_AREA = config['MAX_COMPONENT_AREA']
COMPONENT_CONNECTIVITY = config['COMPONENT_CONNECTIVITY']
K = config['K']
MAX_DISTANCE = config['MAX_DISTANCE']
DISTANCE_METRIC = config['DISTANCE_METRIC']
DATABASE_PATH = config['DATABASE_PATH']

FEATURES = ['percent_fill', 'aspect_ratio', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7']


def threshold(image: cv2.Mat) -> cv2.Mat:
    """
    Applies Otsu's thresholding technique to a given image after converting it to grayscale and applying bilateral filtering.

    :param image: The input image in BGR
    :return: A binary thresholded image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)  # Bilateral filtering to reduce noise while preserving edges
    _, otsu_thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return otsu_thresh


def morph(binary_image: cv2.Mat) -> cv2.Mat:
    """
    Perform morphological closing on a binary image. Closes small holes and joins nearby regions

    :param binary_image: Input binary image
    :return: Morphologically closed binary image
    """
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    return morphed


class ConnectedComponents:
    """
    Class that processes and filters connected components in a binary image.

    :ivar labels: List of valid component labels after filtering based
        on area constraints.
    :ivar label_map: Matrix representing the labeled regions after filtering.
    :ivar stats: Statistics of the connected components
    :ivar centroids: Centroids of the connected components.
    :ivar image: Visualization of the filtered components using random colors.
    """

    def __init__(self, binary_image: cv2.Mat,
                 area_limits: tuple[int, int] = (MIN_COMPONENT_AREA, MAX_COMPONENT_AREA)) -> None:
        """
        Initialize the instance with the filtered connected components from the given binary image

        :param binary_image: Binary image where connected components are computed
        :param area_limits: Tuple specifying the minimum and maximum area of connected components
                            to keep. Defaults to (MIN_COMPONENT_AREA, MAX_COMPONENT_AREA).
        """
        output = cv2.connectedComponentsWithStats(binary_image, connectivity=COMPONENT_CONNECTIVITY, ltype=cv2.CV_32S)
        num_labels, labels, stats, centroids = output

        # Filter components by area
        areas = stats[1:, cv2.CC_STAT_AREA]  # Areas of all components except background
        valid_components = [i + 1 for i, area in enumerate(areas) if area_limits[0] <= area <= area_limits[1]]

        # Create a new labels matrix with only the valid components
        filtered_labels = np.zeros_like(labels)
        for label in valid_components:
            if not (labels == label).any():
                raise ValueError(f"No pixels found for label {label}")
            filtered_labels[labels == label] = label

        # Visualize the components with random colors
        color_map = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        for label in valid_components:
            color = [random.randint(0, 255) for _ in range(3)]  # Random color for each label
            color_map[filtered_labels == label] = color
        color_map[filtered_labels == 0] = [255, 255, 255]  # Set the background (label = 0) to white

        self.labels = valid_components
        self.label_map = filtered_labels
        self.stats = stats
        self.centroids = centroids
        self.image = color_map


class Object:
    """
    Represents an object detected in a connected component analysis

    :ivar label: Unique label identifier for the object.
    :ivar name: Optional name for the object.
    :ivar bounding_box: Coordinates of the oriented bounding box
    :ivar features: Dictionary containing computed features
    """

    def __init__(self, label: int) -> None:
        """
        :param label: The label representing the object in the connected component label map
        """
        self.label = label
        self.name = None
        self.bounding_box = None
        self.features = {}

    def __repr__(self):
        return f"Object(label={self.label})"

    def __str__(self):
        if self.name:
            return f"Object {self.label} - {self.name}"
        else:
            return f"Object {self.label}"

    def calculate_features(self, connected_components: ConnectedComponents) -> None:
        """
        Calculates and stores features for the object:
        [percent fill, aspect ratio, Hu moments]]

        :param connected_components: The connected component object
        :return: None
        """
        # Get the stats for the object
        area = connected_components.stats[self.label, cv2.CC_STAT_AREA]

        # Create a binary mask for the object
        mask = (connected_components.label_map == self.label).astype(np.uint8)
        y_indices, x_indices = np.where(mask > 0)
        points = np.column_stack((x_indices, y_indices))

        # Calculate moments
        moments = cv2.moments(mask)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        mu20 = moments['mu20'] / moments['m00']
        mu11 = moments['mu11'] / moments['m00']
        mu02 = moments['mu02'] / moments['m00']

        # Calculate orientation angle
        theta = 0.5 * np.arctan(2 * mu11 / (mu20 - mu02))

        # Create rotation matrix
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        # Center the points around the centroid
        centered_points = points - np.array([cx, cy])

        # Rotate the points to align with the principal axes
        rotated_points = np.dot(centered_points, rotation_matrix)

        # Find the min and max to determine the bounding box in the rotated space
        min_x = np.min(rotated_points[:, 0])
        max_x = np.max(rotated_points[:, 0])
        min_y = np.min(rotated_points[:, 1])
        max_y = np.max(rotated_points[:, 1])

        # Calculate the width and height of the bounding box in the rotated space
        dims = np.array([max_x - min_x, max_y - min_y])
        width, height = dims.max(), dims.min()

        # Calculate the four corners of the bounding box in the rotated space
        corners_rotated = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ])

        # Rotate back to the original space
        rotation_matrix_inv = np.array([
            [cos_theta, sin_theta],
            [-sin_theta, cos_theta]
        ])
        corners_original = np.dot(corners_rotated, rotation_matrix_inv)

        # Translate back by adding the centroid coordinates
        corners_original = corners_original + np.array([cx, cy])

        # Rotate the bounding box coordinates back to the original coordinate system
        self.bounding_box = np.int32(corners_original)

        # Calculate oriented aspect ratio
        aspect_ratio = width / height

        # Calculate oriented percent fill
        percent_fill = (area / (width * height))

        # Calculate Hu moments
        hu_moments = cv2.HuMoments(moments).flatten()

        # Store the features in the object
        self.features['aspect_ratio'] = aspect_ratio
        self.features['percent_fill'] = percent_fill
        for i, hu in enumerate(hu_moments):
            self.features[f'hu{i + 1}'] = hu

    def show(self, image: cv2.Mat) -> cv2.Mat:
        """
        Draws a bounding box with a label and name on the provided image

        :param image: The input image where the bounding box and text will be drawn.
        :return: A copy of the input image with the bounding box and text overlaid.
        """
        # Draw the bounding box
        image_out = cv2.polylines(image, [np.array(self.bounding_box, dtype=np.int32)], isClosed=True,
                                  color=(0, 0, 255), thickness=2)

        # Add the label and name text
        if self.name:
            label_text = f"{self.name.capitalize()} | ID: {self.label}"
        else:
            label_text = f"ID: {self.label}"
        cv2.putText(image_out, label_text, (int(self.bounding_box[0][0]), int(self.bounding_box[0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return image_out


class Database:
    """
    This class manages a database of objects with associated features,
    providing functionality to add, save, and match objects using KNN classification.

    :ivar db_path: Path to the database CSV file.
    :ivar df: DataFrame holding the main database, where each row corresponds to an object
    :ivar scaler: StandardScaler instance for scaling feature values before processing.
    :ivar knn: K-Nearest Neighbors classifier for matching objects based on feature similarity.
    :ivar scaled_df: Scaled version of df where feature values are normalized
    """

    def __init__(self, db_path: str = DATABASE_PATH) -> None:
        """
        :param db_path: Path to the CSV database file. If the file does not exist, a new database
                        is created at the specified path.
        """
        self.db_path = db_path
        try:
            self.df = pd.read_csv(self.db_path)
            print(f'Database loaded from {self.db_path}')
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=['object'] + FEATURES)
            print(f'No database found. Created new database at {self.db_path}')
        self.scaler = StandardScaler()  # Initialize scaler
        self.knn = KNeighborsClassifier(n_neighbors=K, metric=DISTANCE_METRIC)  # Initialize KNN classifier
        self.scaled_df = self.df.copy()
        # Scale the features and fit the KNN classifier if there are any objects in the database
        if len(self.df) > 0:
            self.scaled_df[FEATURES] = self.scaler.fit_transform(self.scaled_df[FEATURES])
            self.knn.fit(self.scaled_df[FEATURES], self.df['object'])

    def add(self, new_object: Object) -> None:
        """
        Adds a new object to the existing database and updates the scaled database and KNN classifier accordingly.

        :param new_object: The object to be added to the database
        :return: None
        """
        new_entry = {"object": new_object.name}
        for feature in FEATURES:
            new_entry[feature] = new_object.features[feature]
        # Add object to database and scaled database
        self.df = pd.concat([self.df, pd.DataFrame([new_entry])], ignore_index=True)
        self.scaled_df = self.df.copy()
        self.scaled_df[FEATURES] = self.scaler.fit_transform(self.df[FEATURES])
        # Refit KNN classifier with new data
        self.knn.fit(self.scaled_df[FEATURES], self.df['object'])
        print(f'Database updated: {new_object.name} added')

    def save(self) -> None:
        """
        Saves the DataFrame to a CSV file at the specified path.

        :return: None
        """
        self.df.to_csv(self.db_path, index=False)
        print(f'Database saved to {self.db_path}')

    def match(self, objects: list[Object]) -> None:
        """
        Matches a list of objects to existing objects in the database using KNN classification.
        Updates the object attributes if it is matched

        :param objects: List of objects to be matched.
        :return: None
        """

        if len(self.df) > K:  # If there are enough examples for matching
            # Convert list of objects to a DataFrame
            knn_input = pd.DataFrame(columns=FEATURES)
            for obj in objects:
                knn_input.loc[len(knn_input)] = [obj.features[feature] for feature in FEATURES]
            if len(knn_input) == 0:  # No objects to match
                return

            # Scale input features and retain feature names
            knn_input_scaled = pd.DataFrame(self.scaler.transform(knn_input), columns=FEATURES)

            # Get distances and indices of the nearest neighbors
            distances, indices = self.knn.kneighbors(knn_input_scaled)
            for i, obj in enumerate(objects):
                if distances[i].mean() <= MAX_DISTANCE:  # Assign label only if object has nearby neighbors
                    obj.name = self.df['object'].iloc[indices[i][0]]


class Frame:
    """
    Represents a frame of the video feed.
    Contains methods for processing the image, extracting features, and matching objects.

    :ivar original: The original input image.
    :ivar threshold: Binary thresholded version of the input image.
    :ivar morphed: Result of applying morphological operations to the thresholded image.
    :ivar connected_components: Instance of ConnectedComponents with data about the connected component in the frame.
    :ivar objects: List of Object instances
    """

    def __init__(self, image: cv2.Mat) -> None:
        """
        :param image: The input image to be processed
        """
        self.original = image
        self.threshold = threshold(image)
        self.morphed = morph(self.threshold)
        self.connected_components = ConnectedComponents(self.morphed)
        self.objects = [Object(label) for label in self.connected_components.labels]

    def show(self, database: Database, training: bool = False) -> cv2.Mat:
        """
        Displays the processed image highlighting the specific objects depending
        on the mode (training or detection)

        :param database: Database containing object information
        :param training: Flag indicating whether the operation is in training mode
        :return: Annotated image showing detection objects
        """
        image_out = self.original.copy()
        # If in training mode, show all objects else only show named objects
        if training:
            for obj in self.objects:
                image_out = obj.show(image_out)
        else:
            for obj in self.objects:
                if obj.name is not None:
                    image_out = obj.show(image_out)
        return image_out

    def calculate_features(self) -> None:
        """
        Calculates features for each object in the list of objects

        :return: None
        """
        for obj in self.objects:
            obj.calculate_features(self.connected_components)

    def match(self, database: Database) -> None:
        """
        Matches objects identified to objects in the databse

        :return: None
        """
        database.match(self.objects)

    def label_new_object(self, label: int, name: str) -> Optional[Object]:

        """
        Assigns a name to an object given its label

        :param label: The ID of the object
        :param name: The name to be assigned to the object
        :return: The newly named object if found else None
        """
        for obj in self.objects:
            if obj.label == label:
                obj.name = name
                return obj
        print(f'Object not found for label: {label}')
        return None
