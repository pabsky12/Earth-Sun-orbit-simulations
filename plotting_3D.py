import plotly.graph_objects as go

def plot_orbits(positions, body_names, simulator, num_steps, stride, simulate=False):
    """positions: shape (steps+1, n_bodies, 3)"""
    fig = go.Figure()
    for i, name in enumerate(body_names):
        fig.add_trace(go.Scatter3d(
            x=positions[:, i, 0], y=positions[:, i, 1], z=positions[:, i, 2],
            mode="lines", name=name, line=dict(width=3),
        ))
        fig.add_trace(go.Scatter3d(
            x=[positions[0, i, 0]], y=[positions[0, i, 1]], z=[positions[0, i, 2]],
            mode="markers", marker=dict(size=5), showlegend=False,
        ))
    if simulate:
        frames = [
            go.Frame(data=[go.Scatter3d(x=positions[:k,i,0], y=positions[:k,i,1], z=positions[:k,i,2])
                        for i in range(len(body_names))], name=str(k))
            for k in range(0, num_steps, stride)
        ]
        fig.frames = frames
        fig.update_layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])])

        fig.update_layout(
            scene=dict(xaxis_title="x (AU)", yaxis_title="y (AU)", zaxis_title="z (AU)",
                        aspectmode="data"),
            title=f"Orbit paths for {simulator} method",
        )
    return fig


def add_barycenter_trace(fig, barycenter_positions):
    """barycenter_positions: shape (steps+1, 3)"""
    fig.add_trace(go.Scatter3d(
        x=barycenter_positions[:, 0], y=barycenter_positions[:, 1], z=barycenter_positions[:, 2],
        mode="lines", name="Barycenter", line=dict(width=4, dash="dot", color="black"),
    ))
    return fig