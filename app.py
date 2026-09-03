"""
POD-FCDNN Streamlit Web Application
Interactive dashboard for POD-based surrogate modeling of fluid dynamics.
"""

import os
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import streamlit as st

from engine import load_checkpoint, predict_and_reconstruct

# Page Configuration
st.set_page_config(
    page_title="FluidPulse AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ FluidPulse AI")
st.markdown(
    """
    Rapid CFD surrogate modeling power powered by POD-FCDNN neural architectures.
    Real-time flow field prediction and spatial reconstruction.
    """
)

# SIDEBAR: CONTROLS & PARAMETERS
with st.sidebar:
    st.header("Control Panel")

    # CASE SELECTION
    case = st.selectbox(
        "Select Case", ["Cavity", "Cylinder", "Backward Facing Step", "NACA0012"]
    )

    # PARAMETER INPUT
    if case == "NACA0012":
        param = st.slider(
            "Angle of Attack (α)",
            min_value=-5.0,
            max_value=15.0,
            value=0.0,
            step=0.5,
        )
    else:
        param = st.slider(
            "Reynolds Number", min_value=100, max_value=10000, value=1000, step=100
        )

    # PREDICT BUTTON
    predict_btn = st.button("Predict Flow Field", use_container_width=True)


# LOAD CHECKPOINT WITH CACHING FOR MAXIMUM SPEED
@st.cache_resource
def get_model(case_name):
    checkpoint_filenames = {
        "Cavity": "cavity_checkpoint.pt",
        "Cylinder": "cylinder_checkpoint.pt",
        "Backward Facing Step": "bfs_checkpoint.pt",
        "NACA0012": "naca_checkpoint.pt",
    }

    filename = checkpoint_filenames[case_name]
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Check potential path locations
    possible_paths = [
        os.path.join(base_dir, "checkpoints", filename),
        os.path.join(base_dir, filename),
        filename,
    ]

    target_path = None
    for path in possible_paths:
        if os.path.exists(path):
            target_path = path
            break

    if not target_path:
        raise FileNotFoundError(
            f"Checkpoint file '{filename}' not found in 'checkpoints/' or repository root."
        )

    return load_checkpoint(target_path)


# MAIN DASHBOARD AREA
if predict_btn:
    try:
        trainer = get_model(case)

        result = predict_and_reconstruct(trainer, param)

        u = result["u"]
        v = result["v"]
        p = result["p"]
        xy = result["xy"]

        x_coords = xy[:, 0]
        y_coords = xy[:, 1]

        # Calculate tight physical bounds to eliminate excess whitespace
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Interpolate spatial points onto a dense regular grid for smooth rendering
        grid_x_1d = np.linspace(x_min, x_max, 250)
        grid_y_1d = np.linspace(y_min, y_max, 250)
        grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

        grid_p = griddata((x_coords, y_coords), p, (grid_x, grid_y), method="cubic")
        grid_u = griddata((x_coords, y_coords), u, (grid_x, grid_y), method="cubic")
        grid_v = griddata((x_coords, y_coords), v, (grid_x, grid_y), method="cubic")

        st.success(f"Prediction completed for {case}")

        # Helper function for tight, clean flow field figures
        def create_flow_figure(title, z_data, colorscale):
            fig = go.Figure(
                data=go.Contour(
                    x=grid_x_1d,
                    y=grid_y_1d,
                    z=z_data,
                    colorscale=colorscale,
                    line_smoothing=1.3,
                    contours=dict(coloring="heatmap", showlines=False),
                    colorbar=dict(len=0.8, thickness=15),
                )
            )
            fig.update_layout(
                title=dict(text=title, x=0.5, xanchor="center"),
                xaxis=dict(
                    title="x",
                    range=[x_min, x_max],
                    constrain="domain",
                ),
                yaxis=dict(
                    title="y",
                    range=[y_min, y_max],
                    scaleanchor="x",
                    scaleratio=1,
                    constrain="domain",
                ),
                margin=dict(l=10, r=10, t=40, b=10),
                height=450,
            )
            return fig

        # 3-COLUMN SIDE-BY-SIDE LAYOUT
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Absolute Pressure")
            fig_p = create_flow_figure("Pressure Field", grid_p, "Viridis")
            st.plotly_chart(fig_p, use_container_width=True)

        with col2:
            st.subheader("U Velocity")
            fig_u = create_flow_figure("U Velocity Field", grid_u, "RdBu_r")
            st.plotly_chart(fig_u, use_container_width=True)

        with col3:
            st.subheader("V Velocity")
            fig_v = create_flow_figure("V Velocity Field", grid_v, "RdBu_r")
            st.plotly_chart(fig_v, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
else:
    st.info("Select parameters on the left sidebar and click **Predict Flow Field** to run the simulation.")
