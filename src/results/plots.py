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



def plot_bleu_evaluation_metrics(results, model_names, results_dir):
    # Define colors
    local_color = "#006795"
    global_color = "#009B55"

    # Get the top-k and top-p values
    top_k_values = results[model_names[0]]["top_k"]["top_k"].tolist()
    top_p_values = results[model_names[0]]["top_p"]["top_p"].tolist()

    # Create subplots for top-k and top-p
    fig_top_k = make_subplots(
        rows=1,
        cols=4,
        shared_xaxes=True,
        horizontal_spacing=0.035,
        vertical_spacing=0,
        subplot_titles=tuple(model_names),
    )
    fig_top_p = make_subplots(
        rows=1,
        cols=4,
        shared_xaxes=True,
        horizontal_spacing=0.035,
        vertical_spacing=0,
        subplot_titles=tuple(model_names),
    )

    for idx, model_name in enumerate(results.keys(), start=1):
        top_k_df = results[model_name]["top_k"]
        top_p_df = results[model_name]["top_p"]

        fig_top_k.add_trace(
            go.Scatter(
                x=top_k_values,
                y=top_k_df["bleu_local"].tolist(),
                mode="lines+markers",
                name=f"Local Decoding ({model_name})",
                marker=dict(symbol="circle"),
                line=dict(color=local_color),
            ),
            row=1,
            col=idx,
        )
        fig_top_k.add_trace(
            go.Scatter(
                x=top_k_values,
                y=top_k_df["global_bleu"].tolist(),
                mode="lines+markers",
                name=f"Global Decoding ({model_name})",
                marker=dict(symbol="circle"),
                line=dict(color=global_color),
            ),
            row=1,
            col=idx,
        )

        fig_top_p.add_trace(
            go.Scatter(
                x=top_p_values,
                y=top_p_df["bleu_local"].tolist(),
                mode="lines+markers",
                name=f"Local Decoding ({model_name})",
                marker=dict(symbol="circle"),
                line=dict(color=local_color),
            ),
            row=1,
            col=idx,
        )
        fig_top_p.add_trace(
            go.Scatter(
                x=top_p_values,
                y=top_p_df["global_bleu"].tolist(),
                mode="lines+markers",
                name=f"Global Decoding ({model_name})",
                marker=dict(symbol="circle"),
                line=dict(color=global_color),
            ),
            row=1,
            col=idx,
        )

    # Update x-axis properties
    fig_top_k.update_xaxes(title_text="Top-k values", type="category")
    fig_top_p.update_xaxes(title_text="Top-p values", type="category")

    # Get the highest score for the y-axis range
    max_top_k_score = max(
        [
            score
            for model in results.values()
            for score in model["top_k"]["bleu_local"].tolist()
            + model["top_k"]["global_bleu"].tolist()
        ]
    )
    max_top_p_score = max(
        [
            score
            for model in results.values()
            for score in model["top_p"]["bleu_local"].tolist()
            + model["top_p"]["global_bleu"].tolist()
        ]
    )

    # Update y-axis properties to ensure they have the same range
    fig_top_k.update_yaxes(
        range=[0, max_top_k_score], title_text="Self-BLEU Score", row=1, col=1
    )
    fig_top_k.update_yaxes(range=[0, max_top_k_score], row=1, col=2)
    fig_top_k.update_yaxes(range=[0, max_top_k_score], row=1, col=3)
    fig_top_k.update_yaxes(range=[0, max_top_k_score], row=1, col=4)

    fig_top_p.update_yaxes(
        range=[0, max_top_p_score], title_text="Self-BLEU Score", row=1, col=1
    )
    fig_top_p.update_yaxes(range=[0, max_top_p_score], row=1, col=2)
    fig_top_p.update_yaxes(range=[0, max_top_p_score], row=1, col=3)
    fig_top_p.update_yaxes(range=[0, max_top_p_score], row=1, col=4)

    # Update layout
    fig_top_k.update_layout(
        height=360,
        width=1400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.6, xanchor="center", x=0.5),
    )
    fig_top_k.update_xaxes(
        mirror=True, ticks="outside", showline=True, gridcolor="lightgrey"
    )
    fig_top_k.update_yaxes(
        mirror=True, ticks="outside", showline=True, gridcolor="lightgrey"
    )

    fig_top_k.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    fig_top_p.update_layout(
        height=360,
        width=1400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.6, xanchor="center", x=0.5),
    )
    fig_top_p.update_xaxes(
        mirror=True, ticks="outside", showline=True, gridcolor="lightgrey"
    )
    fig_top_p.update_yaxes(
        mirror=True, ticks="outside", showline=True, gridcolor="lightgrey"
    )

    # Save the figure as an EPS file
    fig_top_k.write_image(os.path.join(results_dir, "bleu_top_k.pdf"), "pdf")
    fig_top_k.write_html(os.path.join(results_dir, "bleu_top_k.html"), "html")

    fig_top_p.write_image(os.path.join(results_dir, "bleu_top_p.pdf"), "pdf")
    fig_top_p.write_html(os.path.join(results_dir, "bleu_top_p.html"), "html")

