import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from utils import generate_polygon, generate_windpark, position_to_grid, relative_ws_unnormalization
from models.UNet5 import UNet5 as UNet

def calculate_ct(wind_speed, turbine_rated_ws, turbine_ct):
    if wind_speed <= turbine_rated_ws:
        return turbine_ct
    else:
        return turbine_ct * (turbine_rated_ws / wind_speed) ** 2

st.title("TTE inter-farm kick-off meeting")
st.write("""
This demonstration showcases a workflow for generating random wind farm layouts and simulating wake effects using a neural network.
The process involves three main steps:
1. Generating a random wind farm layout within a polygon
2. Rasterizing the wind farm layout
3. Simulating wake effects using a U-Net neural network
""")
# Create a session state to store the layout parameters and generated data
if 'layout_params' not in st.session_state:
    st.session_state.layout_params = {
        'turbine_amount': 50,
        'turbine_spacing': 8.0,
        'turbine_position_noise': 0.1,
        'polygon_irregularities': 0.2,
        'polygon_spikiness': 0.2,
        'polygon_num_vertices': 6
    }
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = {
        'polygon': None,
        'wind_park': None,
        'grid': None
    }
if 'first_run' not in st.session_state:
    st.session_state.first_run = True

st.header("1. Wind Farm Layout Generation")
st.write("Adjust the parameters below to generate different wind farm layouts:")

col1, col2 = st.columns(2)
# First column of sliders
with col1:
    turbine_amount = st.slider("Turbine Amount", 1, 400, st.session_state.layout_params['turbine_amount'])
    turbine_spacing = st.slider("Turbine Spacing", 6.0, 12.0, st.session_state.layout_params['turbine_spacing'], 0.1)
    turbine_position_noise = st.slider("Turbine Position Noise", 0.0, 0.5, st.session_state.layout_params['turbine_position_noise'], 0.01)

# Second column of sliders
with col2:
    polygon_irregularities = st.slider("Polygon Irregularities", 0.0, 0.5, st.session_state.layout_params['polygon_irregularities'], 0.01)
    polygon_spikiness = st.slider("Polygon Spikiness", 0.0, 0.5, st.session_state.layout_params['polygon_spikiness'], 0.01)
    polygon_num_vertices = st.slider("Polygon Num Vertices", 3, 15, st.session_state.layout_params['polygon_num_vertices'])

def update_layout():
    st.session_state.layout_params = {
        'turbine_amount': turbine_amount,
        'turbine_spacing': turbine_spacing,
        'turbine_position_noise': turbine_position_noise,
        'polygon_irregularities': polygon_irregularities,
        'polygon_spikiness': polygon_spikiness,
        'polygon_num_vertices': polygon_num_vertices
    }
    
    polygon_area = st.session_state.layout_params['turbine_amount'] * st.session_state.layout_params['turbine_spacing']**2
    stats = {
        "polygon_area": polygon_area,
        "turbine_spacing": st.session_state.layout_params['turbine_spacing'],
        "turbine_position_noise": st.session_state.layout_params['turbine_position_noise'],
        "polygon_irregularities": st.session_state.layout_params['polygon_irregularities'],
        "polygon_spikinesses": st.session_state.layout_params['polygon_spikiness'],
        "polygon_num_vertices": st.session_state.layout_params['polygon_num_vertices']
    }
    
    x_range = np.linspace(-256, 768, 512)
    y_range = np.linspace(-256, 256, 256)
    
    polygon = generate_polygon(stats)
    wind_park, polygon = generate_windpark(polygon, stats)
    grid = position_to_grid(x_range, y_range, wind_park)
    
    st.session_state.generated_data['polygon'] = polygon
    st.session_state.generated_data['wind_park'] = wind_park
    st.session_state.generated_data['grid'] = grid

# Add a button to update the layout
if st.button("Update Layout", key="update_layout_button") or st.session_state.first_run:
    update_layout()
    st.session_state.first_run = False

# Use the stored generated data for plotting
polygon = st.session_state.generated_data['polygon']
wind_park = st.session_state.generated_data['wind_park']
grid = st.session_state.generated_data['grid']


if wind_park is not None and polygon is not None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wind_park[0, :],
        y=wind_park[1, :],
        mode='markers',
        marker=dict(size=10, color='blue', opacity=0.6),
        name='Turbines'
    ))

    x, y = polygon
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='red', width=2),
        name='Polygon Boundary'
    ))

    # Update layout
    fig.update_layout(
        title="Wind Park Layout",
        xaxis_title="X/D",
        yaxis_title="Y/D",
        showlegend=True,
        width=700,
        height=700,
        xaxis=dict(range=[-128, 0], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-256, 256])
    )

    # Display the plot
    st.plotly_chart(fig)
    st.write("This plot shows the generated wind farm layout. Blue dots represent wind turbines, and the red line outlines the polygon boundary of the wind farm.")

st.header("2. Wind Farm Rasterization")

if grid is not None:
    fig = go.Figure()
    heatmap = go.Heatmap(
        z=grid,
        colorscale='Viridis',  # You can change this to any colorscale you prefer
        zmin=np.min(grid),
        zmax=np.max(grid),
        showscale=True,  # This will show the color scale bar
        colorbar=dict(title='Bool'),  # Add a title to the color bar
    )
    fig.add_trace(heatmap)

    fig.update_layout(
        title="Wind Park Layout",
        xaxis_title="X/D",
        yaxis_title="Y/D",
        showlegend=True,
        width=700,
        height=350,
    )

    # Display the plot
    st.plotly_chart(fig)
    st.write("This heatmap represents the rasterized wind farm layout. The colored areas indicate the presence of wind turbines.")

st.header("3. Turbine Parameters and Wake Simulation")
st.write("Adjust the turbine parameters and inflow conditions:")

col1, col2 = st.columns(2)
with col1:
    turbine_rated_ws = st.slider("Rated Wind Speed (m/s)", min_value=5.0, max_value=20.0, value=12.0, step=0.1)
with col2:
    turbine_ct = st.slider("Maximum CT", min_value=0.1, max_value=1.0, value=0.8, step=0.01)

wind_speeds = np.linspace(4, 25, 300)
ct_values = [calculate_ct(ws, turbine_rated_ws, turbine_ct) for ws in wind_speeds]

fig = go.Figure()
fig.add_trace(go.Scatter(x=wind_speeds, y=ct_values, mode='lines', name='CT Curve'))
fig.update_layout(
    title=f"CT Curve",
    xaxis_title="Wind Speed [m/s]",
    yaxis_title="Thrust Coefficient [-]",
    yaxis_range=[0, 1]
)
st.plotly_chart(fig)
st.write("This plot shows the Thrust Coefficient (CT) curve for the selected turbine parameters. The CT value affects the intensity of the wake effect.")

st.subheader("Inflow Conditions")
col1, col2 = st.columns(2)
with col1:
    wind_speed = st.slider("Wind Speed", 4.0, 25.0, 11.0, 0.1)
with col2:
    turbulence_intensity = st.slider("Turbulence Intensity", 0.0, 0.3, 0.15, 0.01)


device = "cpu"
dtype = torch.float32
loaded_data = torch.load("./models/model_UNet5_0.0001_e1000_2024-10-05 02:29:57.pth", 
                         map_location=torch.device(device), 
                         weights_only = False)
model = UNet(1, 1, 4).to(device)
state_dict = loaded_data["model_state_dict"]
model.load_state_dict(state_dict)
model.eval()

min_labels = torch.tensor([4.0002, 0.0502, 0.1004, 5.0080]) 
max_labels = torch.tensor([24.9848,  0.2996,  0.9990, 14.9833])

grid_tensor = torch.tensor(grid).unsqueeze(0).unsqueeze(0).to(device, dtype)
labels = torch.tensor([wind_speed, turbulence_intensity, turbine_ct, turbine_rated_ws])
labels = (labels - min_labels) / (max_labels - min_labels).unsqueeze(0).to(device, dtype)

with torch.no_grad():
    output = model(grid_tensor, labels).squeeze()
    output = relative_ws_unnormalization(output, wind_speed)
output = np.array(output)

fig = go.Figure()
contour = go.Heatmap(
    z=output,
    colorscale='Jet',  # You can change this to any colorscale you prefer
    zmin=np.percentile(output, 0.95),
    zmax=np.percentile(output, 0.05),
    showscale=True,  # This will show the color scale bar
    colorbar=dict(title='Wind Speed [m/s]'),  # Add a title to the color bar
)
fig.add_trace(contour)

fig.update_layout(
    title="Wind Park Wake Deficit",
    xaxis_title="X/D",
    yaxis_title="Y/D",
    showlegend=True,
    width=700,
    height=350,
)
st.plotly_chart(fig)
st.write("This heatmap shows the simulated wake deficit across the wind farm. The colors represent the wind speed at different locations, with cooler colors indicating lower wind speeds due to wake effects.")


st.subheader("Neural Network Architecture")
st.image("./images/u_net.png", caption="U-Net neural network")
st.write("The U-Net architecture is used to simulate the wake effects. This neural network takes the rasterized wind farm layout and inflow conditions as input and predicts the wake deficit across the wind farm area.")


st.write("""
In conclusion, this demonstration showcases a complete workflow for wind farm layout generation and wake effect simulation:
1. We generate a random wind farm layout within a polygon.
2. The layout is then rasterized for input into the neural network.
3. Finally, we use a U-Net neural network to simulate the wake effects based on the layout and inflow conditions.

This tool can be valuable for quick assessments of different wind farm layouts and their potential wake interactions.""")