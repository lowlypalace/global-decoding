import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.figure_factory as ff

from scipy.stats import gaussian_kde

from utils import (
    create_filename,
)


def plot_mcmc_distribution(samples, plot_type="histogram", show=True):
    if plot_type == "histogram":
        # Create the histogram data
        trace = go.Histogram(
            x=samples,
            histnorm="probability",
            nbinsx=30,
        )
        layout = go.Layout(
            title="Probability Distribution of MCMC Samples",
            xaxis=dict(title="Sample Value"),
            yaxis=dict(title="Probability Density"),
            bargap=0.2,
        )
        fig = go.Figure(data=[trace], layout=layout)
    elif plot_type == "kde":
        # Generate a kernel density estimate
        # Using Plotly's figure factory to create the KDE plot
        fig = ff.create_distplot(
            [samples],
            group_labels=["KDE"],
            bin_size=0.2,
            show_hist=False,
            show_rug=False,
        )
        fig.update_layout(
            title="Kernel Density Estimate of MCMC Samples",
            xaxis=dict(title="Sample Value"),
            yaxis=dict(title="Density"),
        )
    else:
        raise ValueError("Invalid plot_type. Use 'histogram' or 'kde'.")
    # Write the plot to an HTML file
    fig.write_html(create_filename(f"mcmc_{plot_type}", "html"))
    # Plot the figure
    if show:
        fig.show()


def compute_kde(samples, x_range=None):
    if x_range is None:
        x_range = min(samples), max(samples)
    g_kde = gaussian_kde(samples)
    x_kde = np.linspace(*x_range, num=100)

    return x_kde, g_kde(x_kde)


def plot_chain(
    samples,
    burnin=0.2,
    initial=0.01,
    nsig=1,
    fmt="-",
    y_range=None,
    width=1600,
    height=800,
    margins={"l": 20, "r": 20, "t": 50, "b": 20},
    show=True,
):

    plasma = [
        "rgb(13, 8, 135, 1.0)",
        "rgb(70, 3, 159, 1.0)",
        "rgb(114, 1, 168, 1.0)",
        "rgb(156, 23, 158, 1.0)",
        "rgb(189, 55, 134, 1.0)",
        "rgb(216, 87, 107, 1.0)",
        "rgb(237, 121, 83, 1.0)",
        "rgb(251, 159, 58, 1.0)",
        "rgb(253, 202, 38, 1.0)",
        "rgb(240, 249, 33, 1.0)",
    ]

    num_samples = len(samples)

    idx_burnin = int(num_samples * burnin)
    idx_initial = int(num_samples * initial) + 1

    sample_steps = np.arange(num_samples)

    window = int(0.2 * num_samples)
    df = pd.DataFrame(samples, columns=["samples"])
    df["low_q"] = (
        df["samples"]
        .rolling(window=window, center=True, min_periods=0)
        .quantile(quantile=0.05)
    )
    df["high_q"] = (
        df["samples"]
        .rolling(window=window, center=True, min_periods=0)
        .quantile(quantile=0.95)
    )

    estimate = np.mean(samples)
    stddev = np.std(samples)
    title = f"The estimate over the chain is: {estimate:0.2f} ± {stddev:0.2f}"

    samples_posterior = samples[idx_burnin:]
    samples_burnin = samples[:idx_burnin]
    samples_initial = samples[:idx_initial]

    if y_range is None:
        std_post = np.std(samples_posterior)
        y_range = min(samples) - nsig * std_post, max(samples) + nsig * std_post

    x_kde_posterior, y_kde_posterior = compute_kde(samples_posterior)
    x_kde_burnin, y_kde_burnin = compute_kde(samples_burnin, x_range=y_range)
    x_kde_initial, y_kde_initial = compute_kde(samples_initial, x_range=y_range)

    kde_trace_posterior = go.Scatter(
        x=y_kde_posterior,
        y=x_kde_posterior,
        mode="lines",
        line={"color": plasma[4], "width": 2},
        name="Posterior Distribution",
        xaxis="x2",
        yaxis="y2",
        fill="tozerox",
        fillcolor="rgba(100, 0, 100, 0.20)",
    )

    kde_trace_burnin = go.Scatter(
        x=y_kde_burnin,
        y=x_kde_burnin,
        mode="lines",
        line={"color": plasma[6], "width": 2},
        name="Burnin Distribution",
        xaxis="x2",
        yaxis="y2",
        fill="tozerox",
        fillcolor="rgba(100, 0, 100, 0.20)",
    )

    kde_trace_initial = go.Scatter(
        x=y_kde_initial,
        y=x_kde_initial,
        mode="lines",
        line={"color": plasma[1], "width": 2},
        name="Initial Distribution",
        xaxis="x2",
        yaxis="y2",
        fill="tozerox",
        fillcolor="rgba(100, 0, 100, 0.20)",
    )

    plots = [
        kde_trace_initial,
        kde_trace_burnin,
        kde_trace_posterior,
        go.Scatter(
            x=sample_steps,
            y=df["low_q"],
            line={"color": "rgba(255, 0, 0, 0.0)"},
            showlegend=False,
        ),
        # fill between the endpoints of this trace and the endpoints of the trace before it
        go.Scatter(
            x=sample_steps,
            y=df["high_q"],
            line={"color": "rgba(255, 0, 0, 0.0)"},
            fill="tonextx",
            fillcolor="rgba(100, 0, 100, 0.20)",
            name="Quantile 1 - 99% Region",
        ),
        go.Scatter(
            x=sample_steps[idx_burnin:],
            y=samples_posterior,
            name="Posterior Distribution",
            line={"color": plasma[4]},
        ),
        go.Scatter(
            x=sample_steps[:idx_burnin],
            y=samples_burnin,
            name="Burn-in Region",
            line={"color": plasma[6]},
        ),
        go.Scatter(
            x=sample_steps[:idx_initial],
            y=samples_initial,
            name="Initial Condition Dominated Region",
            line={"color": plasma[1]},
        ),
    ]

    layout = go.Layout(
        title=title,
        xaxis={"domain": [0, 0.88], "showgrid": False},
        xaxis2={"domain": [0.9, 1], "showgrid": False},
        yaxis={"range": y_range, "showgrid": False},
        yaxis2={"anchor": "x2", "range": y_range, "showgrid": False},
        width=width,
        height=height,
        margin=margins,
        plot_bgcolor="rgba(255, 255, 255, 1)",
        paper_bgcolor="rgba(255, 255, 255, 1)",
    )

    fig = go.Figure(plots, layout=layout)
    # Write the plot to an HTML file
    fig.write_html(create_filename(f"mcmc_chain", "html"))
    if show:
        fig.show()
