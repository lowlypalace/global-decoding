import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

# To avoid bug in graphs
pio.kaleido.scope.mathjax = None


def plot_sequences_lengths(results, results_dir):
    for model_name, data in results.items():
        top_k_df = data["top_k"]
        top_p_df = data["top_p"]

        # Create subplots for top-k and top-p
        fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.07)
        # subplot_titles=('Average Lengths by top-k', 'Average Lengths by Top-p')

        # Define colors
        local_color = "#006795"
        global_color = "#009B55"

        # Add bar traces for top-k
        fig.add_trace(
            go.Bar(
                x=top_k_df["top_k"],
                width=0.2,
                y=top_k_df["avg_length_local"],
                name="Local Normalisation",
                marker_color=local_color,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=top_k_df["top_k"],
                width=0.2,
                y=top_k_df["avg_length_global"],
                name="Global Normalisation",
                marker_color=global_color,
            ),
            row=1,
            col=1,
        )

        # Add bar traces for top-p
        fig.add_trace(
            go.Bar(
                x=top_p_df["top_p"],
                width=0.2,
                y=top_p_df["avg_length_local"],
                name="Local Normalisation",
                marker_color=local_color,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=top_p_df["top_p"],
                width=0.2,
                y=top_p_df["avg_length_global"],
                name="Global Normalisation",
                marker_color=global_color,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Update x-axis properties
        fig.update_xaxes(title_text="Top-k values", row=1, col=1, type="category")
        fig.update_xaxes(title_text="Top-p values", row=1, col=2, type="category")

        # Update y-axis properties
        fig.update_yaxes(title_text="Average Length", row=1, col=1)
        # fig.update_yaxes(title_text="Average Length", row=1, col=2)

        # Update x-axes properties
        fig.update_xaxes(
            mirror=True, ticks="outside", showline=True, gridcolor="lightgrey"
        )

        # Update y-axes properties
        fig.update_yaxes(
            mirror=True, ticks="outside", showline=True, gridcolor="lightgrey"
        )

        # Update layout background color
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        # Update layout
        fig.update_layout(
            height=360,
            width=1400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5
            ),
        )

        # Save the figure
        fig.write_image(
            os.path.join(results_dir, f"average_lengths_{model_name}.pdf"), "pdf"
        )
        fig.write_html(
            os.path.join(results_dir, f"average_lengths_{model_name}.html"), "html"
        )


def plot_average_log_likelihood(results, results_dir):
    # Create subplots for top-k and top-p
    fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.07)

    # Define colors for each model (adjust as needed)
    colors = {
        "pythia-70m": "#92268F",
        "pythia-410m": "#FBB982",
        "pythia-1.4b": "#41B0E4",
    }

    for model_name, data in results.items():
        top_k_df = data["top_k"]
        top_p_df = data["top_p"]
        color = colors.get(model_name, "#000000")  # Default to black if color not found

        # Add line traces for top-k
        fig.add_trace(
            go.Scatter(
                x=top_k_df["top_k"],
                y=top_k_df["average_log_likelihood"],
                mode="lines+markers",
                name=f"{model_name} (top-k)",
                marker_color=color,
            ),
            row=1,
            col=1,
        )

        # Add line traces for top-p
        fig.add_trace(
            go.Scatter(
                x=top_p_df["top_p"],
                y=top_p_df["average_log_likelihood"],
                mode="lines+markers",
                name=f"{model_name} (top-p)",
                marker_color=color,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Update x-axis properties
    fig.update_xaxes(title_text="Top-k values", row=1, col=1, type="category")
    fig.update_xaxes(title_text="Top-p values", row=1, col=2, type="category")

    # Update y-axis properties
    fig.update_yaxes(title_text="Average Log Likelihood", row=1, col=1)

    # Update x-axes properties
    fig.update_xaxes(mirror=True, ticks="outside", showline=True, gridcolor="lightgrey")

    # Update y-axes properties
    fig.update_yaxes(mirror=True, ticks="outside", showline=True, gridcolor="lightgrey")

    # Update layout background color
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # Update layout
    fig.update_layout(
        height=360,
        width=1400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
    )

    # Save the figure as a PDF file
    fig.write_image(os.path.join(results_dir, "average_log_likelihood.pdf"), "pdf")
    fig.write_html(os.path.join(results_dir, "average_log_likelihood.html"), "html")


def update_fig(fig, max_score, xaxis_title_text, y_axis_title_text):

    fig.update_xaxes(title_text=xaxis_title_text, type="category")
    for col in range(1, 5):
        fig.update_yaxes(range=[0, max_score], row=1, col=col)
    fig.update_yaxes(title_text=y_axis_title_text, row=1, col=1)
    fig.update_layout(
        height=360,
        width=1400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.6, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(mirror=True, ticks="outside", showline=True, gridcolor="lightgrey")
    fig.update_yaxes(mirror=True, ticks="outside", showline=True, gridcolor="lightgrey")


def add_traces(
    fig, values, local_scores, global_scores, local_color, global_color, row, col
):
    fig.add_trace(
        go.Scatter(
            x=values,
            y=local_scores,
            mode="lines+markers",
            name=f"Local Decoding",
            marker=dict(symbol="circle"),
            line=dict(color=local_color),
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=values,
            y=global_scores,
            mode="lines+markers",
            name=f"Global Decoding",
            marker=dict(symbol="circle"),
            line=dict(color=global_color),
        ),
        row=row,
        col=col,
    )


def plot_bleu_evaluation_metrics(results, model_names, results_dir):

    local_color = "#006795"
    global_color = "#009B55"

    top_k_values = results[model_names[0]]["top_k"]["top_k"].tolist()
    top_p_values = results[model_names[0]]["top_p"]["top_p"].tolist()

    fig_top_k = make_subplots(
        rows=1,
        cols=4,
        shared_xaxes=True,
        horizontal_spacing=0.035,
        subplot_titles=tuple(model_names),
    )
    fig_top_p = make_subplots(
        rows=1,
        cols=4,
        shared_xaxes=True,
        horizontal_spacing=0.035,
        subplot_titles=tuple(model_names),
    )

    for idx, model_name in enumerate(results.keys(), start=1):
        top_k_df = results[model_name]["top_k"]
        top_p_df = results[model_name]["top_p"]

        add_traces(
            fig_top_k,
            top_k_values,
            top_k_df["bleu_local"].tolist(),
            top_k_df["bleu_global"].tolist(),
            local_color,
            global_color,
            1,
            idx,
        )
        add_traces(
            fig_top_p,
            top_p_values,
            top_p_df["bleu_local"].tolist(),
            top_p_df["bleu_global"].tolist(),
            local_color,
            global_color,
            1,
            idx,
        )

    max_top_k_score = max(
        score
        for model in results.values()
        for score in model["top_k"]["bleu_local"].tolist()
        + model["top_k"]["bleu_global"].tolist()
    )
    max_top_p_score = max(
        score
        for model in results.values()
        for score in model["top_p"]["bleu_local"].tolist()
        + model["top_p"]["bleu_global"].tolist()
    )

    update_fig(fig_top_k, max_top_k_score, "Top-k values", "Self-BLEU Score")
    update_fig(fig_top_p, max_top_p_score, "Top-p values", "Self-BLEU Score")

    fig_top_k.write_image(os.path.join(results_dir, "bleu_top_k.pdf"), "pdf")
    fig_top_k.write_html(os.path.join(results_dir, "bleu_top_k.html"), "html")

    fig_top_p.write_image(os.path.join(results_dir, "bleu_top_p.pdf"), "pdf")
    fig_top_p.write_html(os.path.join(results_dir, "bleu_top_p.html"), "html")


def plot_mauve_evaluation_metrics(results, model_names, results_dir):

    local_color = "#006795"
    global_color = "#009B55"

    top_k_values = results[model_names[0]]["top_k"]["top_k"].tolist()
    top_p_values = results[model_names[0]]["top_p"]["top_p"].tolist()

    fig_top_k = make_subplots(
        rows=1,
        cols=4,
        shared_xaxes=True,
        horizontal_spacing=0.035,
        subplot_titles=tuple(model_names),
    )
    fig_top_p = make_subplots(
        rows=1,
        cols=4,
        shared_xaxes=True,
        horizontal_spacing=0.035,
        subplot_titles=tuple(model_names),
    )

    for idx, model_name in enumerate(results.keys(), start=1):
        top_k_df = results[model_name]["top_k"]
        top_p_df = results[model_name]["top_p"]

        add_traces(
            fig_top_k,
            top_k_values,
            top_k_df["mauve_local"].tolist(),
            top_k_df["mauve_global"].tolist(),
            local_color,
            global_color,
            1,
            idx,
        )
        add_traces(
            fig_top_p,
            top_p_values,
            top_p_df["mauve_local"].tolist(),
            top_p_df["mauve_global"].tolist(),
            local_color,
            global_color,
            1,
            idx,
        )

    max_top_k_score = max(
        score
        for model in results.values()
        for score in model["top_k"]["mauve_local"].tolist()
        + model["top_k"]["mauve_global"].tolist()
    )
    max_top_p_score = max(
        score
        for model in results.values()
        for score in model["top_p"]["mauve_local"].tolist()
        + model["top_p"]["mauve_global"].tolist()
    )

    update_fig(fig_top_k, max_top_k_score, "Top-k values", "MAUVE Score")
    update_fig(fig_top_p, max_top_p_score, "Top-p values", "MAUVE Score")

    fig_top_k.write_image(os.path.join(results_dir, "mauve_top_k.pdf"), "pdf")
    fig_top_k.write_html(os.path.join(results_dir, "mauve_top_k.html"), "html")

    fig_top_p.write_image(os.path.join(results_dir, "mauve_top_p.pdf"), "pdf")
    fig_top_p.write_html(os.path.join(results_dir, "mauve_top_p.html"), "html")
