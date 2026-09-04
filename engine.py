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
    Rapid CFD surrogate modeling powered by POD-FCDNN neural architectures.
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

    # VARIABLE SELECTION
    selected_variable = st.radio(
        "Select Flow Variable",
        ["All Variables", "Absolute Pressure", "U Velocity", "V Velocity"],
        index=0,
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

        # AUTO-ZOOM & CROP LIMITS SPECIFIC TO GEOMETRY REGIONS OF INTEREST
        if case == "Cylinder":
            x_min, x_max = -1.0, 5.0
            y_min, y_max = -1.5, 1.5
            grid_res = 250
        elif case == "Backward Facing Step":
            x_min, x_max = -1.0, 8.0
            y_min, y_max = -0.5, 1.5
            grid_res = 200  # Optimized resolution for instant BFS processing
        elif case == "NACA0012":
            x_min, x_max = -0.5, 1.8
            y_min, y_max = -0.8, 0.8
            grid_res = 250
        else:
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            grid_res = 250

        # Fast regular grid interpolation
        grid_x_1d = np.linspace(x_min, x_max, grid_res)
        grid_y_1d = np.linspace(y_min, y_max, grid_res)
        grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

        # Switched to 'linear' interpolation for significantly faster execution (especially BFS)
        grid_p = griddata((x_coords, y_coords), p, (grid_x, grid_y), method="linear")
        grid_u = griddata((x_coords, y_coords), u, (grid_x, grid_y), method="linear")
        grid_v = griddata((x_coords, y_coords), v, (grid_x, grid_y), method="linear")

        st.success(f"Prediction completed for {case}")

        # UNIFORM PLOTTING FUNCTION FOR EQUAL FIELD SIZES
        def create_flow_figure(z_data, colorscale="Turbo", height=480):
            fig = go.Figure(
                data=go.Contour(
                    x=grid_x_1d,
                    y=grid_y_1d,
                    z=z_data,
                    colorscale=colorscale,
                    line_smoothing=1.3,
                    contours=dict(
                        coloring="heatmap",
                        showlines=False,
                    ),
                    line=dict(width=0),
                    colorbar=dict(
                        len=0.9,
                        thickness=14,
                        tickfont=dict(size=10),
                    ),
                )
            )
            fig.update_layout(
                xaxis=dict(
                    title="x",
                    range=[x_min, x_max],
                    showgrid=False,
                    zeroline=False,
                    autorange=False,
                ),
                yaxis=dict(
                    title="y",
                    range=[y_min, y_max],
                    showgrid=False,
                    zeroline=False,
                    autorange=False,
                    # Removed scaleanchor="x" constraint so all figures stretch evenly
                ),
                margin=dict(l=15, r=15, t=15, b=15),
                height=height,
                autosize=True,
            )
            return fig

        plotly_config = {"displayModeBar": False}

        # RENDER BASED ON VARIABLE SELECTION
        if selected_variable == "All Variables":
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Absolute Pressure")
                fig_p = create_flow_figure(grid_p, colorscale="Turbo", height=450)
                st.plotly_chart(fig_p, use_container_width=True, config=plotly_config)

            with col2:
                st.subheader("U Velocity")
                fig_u = create_flow_figure(grid_u, colorscale="Turbo", height=450)
                st.plotly_chart(fig_u, use_container_width=True, config=plotly_config)

            with col3:
                st.subheader("V Velocity")
                fig_v = create_flow_figure(grid_v, colorscale="Turbo", height=450)
                st.plotly_chart(fig_v, use_container_width=True, config=plotly_config)

        elif selected_variable == "Absolute Pressure":
            st.subheader("Absolute Pressure")
            fig_p = create_flow_figure(grid_p, colorscale="Turbo", height=650)
            st.plotly_chart(fig_p, use_container_width=True, config=plotly_config)

        elif selected_variable == "U Velocity":
            st.subheader("U Velocity")
            fig_u = create_flow_figure(grid_u, colorscale="Turbo", height=650)
            st.plotly_chart(fig_u, use_container_width=True, config=plotly_config)

        elif selected_variable == "V Velocity":
            st.subheader("V Velocity")
            fig_v = create_flow_figure(grid_v, colorscale="Turbo", height=650)
            st.plotly_chart(fig_v, use_container_width=True, config=plotly_config)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
else:
    st.info("Select parameters on the left sidebar and click **Predict Flow Field** to run the simulation.")
