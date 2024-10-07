# Wind Farm Simulator

## Overview
This Streamlit application demonstrates a comprehensive workflow for generating random wind farm layouts and simulating wake effects using a neural network. The process involves three main steps:

1. Generating a random wind farm layout within a polygon
2. Rasterizing the wind farm layout
3. Simulating wake effects using a U-Net neural network

## Features
- Interactive wind farm layout generation with adjustable parameters
- Visualization of the generated wind farm layout
- Rasterization of the wind farm layout for neural network input
- Customizable turbine parameters and inflow conditions
- Visualization of the simulated wake deficit

## Requirements
- Python 3.x
- NumPy
- Plotly
- Streamlit
- PyTorch
- Custom utility functions (`utils.py`)
- Pre-trained U-Net model (`models/UNet5.py`)

## Installation
1. Clone the repository
2. Install the required packages:
   ```
   pip install numpy plotly streamlit torch
   ```
3. Ensure you have the custom `utils.py` file and the pre-trained model in the correct directories.

## Usage
1. Run the Streamlit app:
   ```
   streamlit run main.py
   ```
2. Use the sliders to adjust wind farm layout parameters, turbine characteristics, and inflow conditions.
3. Click the "Update Layout" button to generate a new wind farm layout.
4. Observe the visualizations for the wind farm layout, rasterization, and wake deficit simulation.