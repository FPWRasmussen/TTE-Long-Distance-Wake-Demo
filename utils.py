import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from py_wake.deficit_models import (BastankhahGaussianDeficit, NOJDeficit,
                                    TurboGaussianDeficit)
from py_wake.deficit_models.deficit_model import (BlockageDeficitModel,
                                                  WakeDeficitModel)
from py_wake.flow_map import HorizontalGrid
from py_wake.site._site import UniformSite
from py_wake.superposition_models import SquaredSum, SuperpositionModel
from py_wake.turbulence_models import TurbulenceModel
from py_wake.utils.model_utils import get_models
from py_wake.wind_farm_models import All2AllIterative, PropagateDownwind
from py_wake.wind_farm_models.wind_farm_model import SimulationResult
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import CubePowerSimpleCt
from shapely.affinity import scale
from shapely.geometry import Point, Polygon


def scale_polygon(polygon, desired_area):
    current_polygon_area = polygon.area
    scale_factor = (desired_area/current_polygon_area)**0.5
    centroid = polygon.centroid
    scaled_polygon = scale(polygon, xfact=scale_factor, yfact=scale_factor, origin=centroid)
    return scaled_polygon

def generate_polygon(stats : dict) -> Polygon:
    def random_angle_steps(steps: int, irregularity: float) -> np.ndarray:
        """Generates the division of a circumference in random angles with a normal distribution.

        Args:
            steps (int):
                the number of angles to generate.
            irregularity (float):
                variance of the spacing of the angles between consecutive vertices.
        Returns:
            np.ndarray: the array of the random angles.
        """
        mean_angle = 2 * np.pi / steps  # mean of the angles

        # Generate n angle steps with a normal distribution
        angles = np.random.normal(loc=mean_angle, scale=irregularity, size=steps)
        angle_sum = np.sum(angles)

        # Normalize the steps so that the sum equals 2π
        angles *= (2 * np.pi) / angle_sum
        return angles

    def clip(value, lower, upper):
        """
        Given an interval, values outside the interval are clipped to the interval edges.
        """
        return np.minimum(upper, np.maximum(value, lower))
    
    irregularity = stats["polygon_irregularities"]
    spikiness = stats["polygon_spikinesses"]
    irregularity *= 2 * np.pi / stats["polygon_num_vertices"]
    angle_steps = random_angle_steps(stats["polygon_num_vertices"], irregularity)
    angles = np.cumsum(angle_steps)
    radii = clip(np.random.normal(1, spikiness, stats["polygon_num_vertices"]), 0.2, 5) # maybe look here
    points = np.column_stack((radii * np.cos(angles),
                              radii * np.sin(angles)))
    
    polygon = scale_polygon(Polygon(points), float(stats["polygon_area"]))
    
    return polygon

def rotate_points_around_centroid(points, angle_degrees, center_of_rotation=None, random_direction: bool = True):
   """
   Rotate a set of 2D points around a specified centroid by a given angle.

   Args:
       points (numpy.ndarray): An array of shape (n, 2) or (2, n) representing the (x, y) coordinates of the points.
       angle_degrees (float): The angle to rotate the points by, in degrees.
       center_of_rotation (list or numpy.ndarray, optional): The (x, y) coordinates of the centroid to rotate around.
           If None (default), the centroid is calculated as the mean of the points.
       random_direction (bool, optional): If True (default), the angle is randomly chosen between -angle_degrees and angle_degrees.

   Returns:
       numpy.ndarray: An array of shape (n, 2) representing the rotated points.

   Example:
       >>> points = np.array([[1, 2], [3, 4], [5, 6]])
       >>> rotated_points = rotate_points_around_centroid(points, 90)
       >>> print(rotated_points)
       [[-2.  1.]
        [-4.  3.]
        [-6.  5.]]
   """
   points = np.array(points)

   # Ensure the points have the correct shape (n, 2)
   if points.shape[0] != 2:
       points = points.T

   angle_radians = np.deg2rad(angle_degrees)  # Convert the angle to radians

   if random_direction:
       angle_radians = np.random.uniform(low=-angle_radians, high=angle_radians)

   # Calculate the centroid if not provided
   if center_of_rotation is None:
       center_of_rotation = np.mean(points, axis=1, keepdims=True)
   else:
       center_of_rotation = np.array(center_of_rotation).reshape(2, 1)

   centered_points = points - center_of_rotation  # Subtract the centroid from each point

   # Create the rotation matrix
   rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                               [np.sin(angle_radians), np.cos(angle_radians)]])

   rotated_centered_points = rotation_matrix @ centered_points  # Apply the rotation matrix
   rotated_points = rotated_centered_points + center_of_rotation  # Add back the centroid

   return rotated_points

def generate_windpark(polygon : Polygon, stats : dict) -> np.ndarray:
    """
    Generate noisy grid points within the bounding box of a polygon.

    Args:
        polygon: Polygon object representing the area of interest.
        spacing (float): Spacing between grid points.

    Returns:
        np.ndarray: Array containing noisy grid points
               inside the specified polygon.
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    
    turbine_spacing = stats["turbine_spacing"]

    x_points = np.arange(min_x, max_x, turbine_spacing)
    y_points = np.arange(min_y, max_y, turbine_spacing)
    X, Y = np.meshgrid(x_points, y_points)
    coordinates = np.column_stack((X.ravel(), Y.ravel()))
    point_mask = np.zeros(len(coordinates), dtype=bool)

    for i, coord in enumerate(coordinates):
        if polygon.contains(Point(coord)):
            point_mask[i] = True

    X_polygon, Y_polygon = np.array(polygon.exterior.xy)


    point_mask = point_mask.reshape(X.shape)
    X_masked, Y_masked = X[point_mask], Y[point_mask]
    X_rotated, Y_rotated = rotate_points_around_centroid(np.array([X_masked, Y_masked]), 0, center_of_rotation=None, random_direction = True)
    X_polygon, Y_polygon = rotate_points_around_centroid(np.array([X_polygon, Y_polygon]), 0, center_of_rotation=None, random_direction = True)
    X_noisy = X_rotated + np.random.normal(loc=0, scale=turbine_spacing * stats["turbine_position_noise"], size=X_masked.shape)
    Y_noisy = Y_rotated + np.random.normal(loc=0, scale=turbine_spacing * stats["turbine_position_noise"], size=Y_masked.shape)

    if X_noisy.size > 0:
        x_max = np.max(X_noisy)
        X_noisy = X_noisy - x_max # Move the wind park to negative coordinates
        X_polygon = X_polygon - x_max
    else:
        return np.vstack([X_noisy, Y_noisy]), np.vstack([X_polygon, Y_polygon])
    
    return np.vstack([X_noisy, Y_noisy]), np.vstack([X_polygon, Y_polygon])

def position_to_grid(x_range, y_range, positions):
    x_range = x_range
    y_range = y_range
    positions = positions

    num_x_cells = len(x_range)
    num_y_cells = len(y_range)

    # Create an empty 2D grid
    grid = np.zeros([num_y_cells, num_x_cells])

    # Iterate over each position
    for x, y in positions.T:
        # Find the nearest grid cell indices for the position
        x_idx = np.argmin(np.abs(x_range - x))
        y_idx = np.argmin(np.abs(y_range - y))

        # Calculate the distance from the position to the center of the nearest grid cell
        x_dist = x - x_range[x_idx]
        y_dist = y - y_range[y_idx]

        # Calculate the contribution of the position to the grid cell and its neighbors
        x_con = 1 - np.abs(x_dist) / (x_range[1] - x_range[0])
        y_con = 1 - np.abs(y_dist) / (y_range[1] - y_range[0])

        if x_dist >= 0 and y_dist >= 0:
            grid[y_idx, x_idx] += x_con * y_con
            grid[y_idx, x_idx + 1] += (1 - x_con) * y_con
            grid[y_idx + 1, x_idx] += x_con * (1 - y_con)
            grid[y_idx + 1, x_idx + 1] += (1 - x_con) * (1 - y_con)
        elif x_dist >= 0 and y_dist < 0:
            grid[y_idx, x_idx] += x_con * y_con
            grid[y_idx, x_idx + 1] += (1 - x_con) * y_con
            grid[y_idx - 1, x_idx] += x_con * (1 - y_con)
            grid[y_idx - 1, x_idx + 1] += (1 - x_con) * (1 - y_con)
        elif x_dist < 0 and y_dist >= 0:
            grid[y_idx, x_idx] += x_con * y_con
            grid[y_idx, x_idx - 1] += (1 - x_con) * y_con
            grid[y_idx + 1, x_idx] += x_con * (1 - y_con)
            grid[y_idx + 1, x_idx - 1] += (1 - x_con) * (1 - y_con)
        elif x_dist < 0 and y_dist < 0:
            grid[y_idx, x_idx] += x_con * y_con
            grid[y_idx, x_idx - 1] += (1 - x_con) * y_con
            grid[y_idx - 1, x_idx] += x_con * (1 - y_con)
            grid[y_idx - 1, x_idx - 1] += (1 - x_con) * (1 - y_con)

    return grid

def relative_ws_unnormalization(x, ws):
    x = ws*(1 - (1- x)**2)
    return x

def calculate_ct(wind_speed, turbine_rated_ws, turbine_ct):
    if wind_speed <= turbine_rated_ws:
        return turbine_ct
    else:
        return turbine_ct * (turbine_rated_ws / wind_speed) ** 2