"""
POD-FCDNN Streamlit Web Application
Interactive dashboard for POD-based surrogate modeling of fluid dynamics.
"""

import streamlit as st
import plotly.graph_objects as go
import os

from engine import (
    load_checkpoint,
    predict_and_reconstruct
)

# Page Configuration
st.set_page_config(
    page_title="POD-FCDNN Surrogate Model",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 POD-FCDNN Fluid Dynamics Surrogate Model")
st.markdown(
    """
    Train and deploy a neural network surrogate model for rapid CFD prediction.
    Combines Proper Orthogonal Decomposition with Deep Learning.
    """
)

st.header("Flow Field Prediction")

# CASE SELECTION
case = st.selectbox(
    "Select Case",
    [
        "Cavity",
        "Cylinder",
        "Backward Facing Step",
        "NACA0012"
    ]
)

# PARAMETER INPUT
if case == "NACA0012":
    param = st.slider(
        "Angle of Attack (α)",
        min_value=-5.0,
        max_value=15.0,
        value=0.0,
        step=0.5
    )
else:
    param = st.slider(
        "Reynolds Number",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )

# LOAD CHECKPOINT WITH CACHING FOR MAXIMUM SPEED
@st.cache_resource
def get_model(case_name):
    checkpoint_paths = {
        "Cavity": "cavity_checkpoint.pt",
        "Cylinder": "cylinder_checkpoint.pt",
        "Backward Facing Step": "bfs_checkpoint.pt",
        "NACA0012": "naca_checkpoint.pt"
    }

    path = checkpoint_paths[case_name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file '{path}' not found in repository root.")

    return load_checkpoint(path)

# PREDICT BUTTON
predict_btn = st.button(
    "Predict Flow Field",
    use_container_width=True
)

if predict_btn:
    try:
        trainer = get_model(case)

        result = predict_and_reconstruct(
            trainer,
            param
        )

        u = result["u"]
        v = result["v"]
        p = result["p"]
        xy = result["xy"]

        x_coords = xy[:, 0]
        y_coords = xy[:, 1]

        st.success(f"Prediction completed for {case}")

        # PRESSURE FIELD
        st.subheader("Pressure Field")
        fig_p = go.Figure(
            data=go.Scattergl(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(
                    size=4,
                    color=p,
                    colorscale="Viridis",
                    showscale=True
                )
            )
        )
        fig_p.update_layout(
            title=f"{case} Pressure Field",
            xaxis_title="x",
            yaxis_title="y",
            height=500
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # U VELOCITY
        st.subheader("U Velocity")
        fig_u = go.Figure(
            data=go.Scattergl(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(
                    size=4,
                    color=u,
                    colorscale="RdBu_r",
                    showscale=True
                )
            )
        )
        fig_u.update_layout(
            title=f"{case} U Velocity",
            xaxis_title="x",
            yaxis_title="y",
            height=500
        )
        st.plotly_chart(fig_u, use_container_width=True)

        # V VELOCITY
        st.subheader("V Velocity")
        fig_v = go.Figure(
            data=go.Scattergl(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(
                    size=4,
                    color=v,
                    colorscale="RdBu_r",
                    showscale=True
                )
            )
        )
        fig_v.update_layout(
            title=f"{case} V Velocity",
            xaxis_title="x",
            yaxis_title="y",
            height=500
        )
        st.plotly_chart(fig_v, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
