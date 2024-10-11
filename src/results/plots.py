import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import numpy as np
import pandas as pd
from plotly.colors import qualitative


# To avoid bug in graphs
pio.kaleido.scope.mathjax = None

# local_color = "#006795"
# global_color = "#009B55"
local_color = "#377EB8"
global_color = "#4DAF4A"

title_font = 16
tick_font = 14


def plot_sequences_lengths(results, results_dir):
    for model_name, data in results.items():
        top_k_df = data["top_k"]
        top_p_df = data["top_p"]

        # Create subplots for top-k and top-p
        fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.07)

        # Add bar traces for top-k with confidence intervals
        fig.add_trace(
            go.Bar(
                name="Local Decoding",
                x=top_k_df["top_k"],
                y=top_k_df["avg_length_local_mean"],
                marker_color=local_color,
                width=0.2,
                error_y=dict(
                    type="data",
                    array=[(ci[1] - ci[0]) / 2 for ci in top_k_df["avg_length_local_ci"]],
                    visible=True,
                    thickness=0.5,
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name="Global Decoding",
                x=top_k_df["top_k"],
                y=top_k_df["avg_length_global_mean"],
                marker_color=global_color,
                width=0.2,
                error_y=dict(
                    type="data",
                    array=[(ci[1] - ci[0]) / 2 for ci in top_k_df["avg_length_global_ci"]],
                    visible=True,
                    thickness=0.5,
                ),
            ),
            row=1,
            col=1,
        )

        # Add bar traces for top-p with confidence intervals
        fig.add_trace(
            go.Bar(
                name="Local Decoding",
                x=top_p_df["top_p"],
                y=top_p_df["avg_length_local_mean"],
                marker_color=local_color,
                width=0.2,
                error_y=dict(
                    type="data",
                    array=[(ci[1] - ci[0]) / 2 for ci in top_p_df["avg_length_local_ci"]],
                    visible=True,
                    thickness=0.5,
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                name="Global Decoding",
                x=top_p_df["top_p"],
                y=top_p_df["avg_length_global_mean"],
                marker_color=global_color,
                width=0.2,
                error_y=dict(
                    type="data",
                    array=[(ci[1] - ci[0]) / 2 for ci in top_p_df["avg_length_global_ci"]],
                    visible=True,
                    thickness=0.5,
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Update x-axis properties
        fig.update_xaxes(title_text="Top-k values", row=1, col=1, type="category")
        fig.update_xaxes(title_text="Top-π values", row=1, col=2, type="category")

        # Update y-axis properties
        fig.update_yaxes(title_text="Average Length", row=1, col=1)

        # Add gridlines back on x and y axes
        fig.update_xaxes(mirror=True, ticks="outside", showline=True, gridcolor="lightgrey")
        fig.update_yaxes(mirror=True, ticks="outside", showline=True, gridcolor="lightgrey")

        # Update layout to include font styles
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=12, color="black"),
            height=360,
            width=1400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
        )

        fig.update_yaxes(
            title_font=dict(size=title_font),
        )

        fig.update_xaxes(
            title_font=dict(size=title_font),
        )

        # Update layout to set legend font size
        fig.update_layout(legend=dict(font=dict(size=title_font)))

        # Save the figure
        fig.write_image(os.path.join(results_dir, f"average_lengths_{model_name}.pdf"), "pdf")
        fig.write_html(os.path.join(results_dir, f"average_lengths_{model_name}.html"), "html")


def plot_average_log_likelihood(results, results_dir, type):
    fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.07)

    colors = {
        "pythia-70m": qualitative.Plotly[0],
        "pythia-410m": qualitative.Plotly[1],
        "pythia-1.4b": qualitative.Plotly[2],
        "pythia-2.8b": qualitative.Plotly[9],
    }

    for model_name, data in results.items():
        top_k_df = data["top_k"]
        top_p_df = data["top_p"]
        color = colors.get(model_name, "#000000")  # Default to black if color not found

        # Add line traces for top-k
        fig.add_trace(
            go.Scatter(
                x=top_k_df["top_k"],
                y=top_k_df[f"log_likelihoods_{type}_mean"],
                mode="lines+markers",
                name=f"{model_name}",
                marker_color=color,
            ),
            row=1,
            col=1,
        )

        # Add line traces for top-p
        fig.add_trace(
            go.Scatter(
                x=top_p_df["top_p"],
                y=top_p_df[f"log_likelihoods_{type}_mean"],
                mode="lines+markers",
                name=f"{model_name}",
                marker_color=color,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Update x-axis properties
    fig.update_xaxes(title_text="Top-k values", row=1, col=1, type="category")
    fig.update_xaxes(title_text="Top-π values", row=1, col=2, type="category")

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

    fig.update_yaxes(
        title_font=dict(size=title_font),
    )

    fig.update_xaxes(
        title_font=dict(size=title_font),
    )

    # Update layout to set legend font size
    fig.update_layout(legend=dict(font=dict(size=title_font)))

    # Save the figure as a PDF file
    fig.write_image(os.path.join(results_dir, f"average_log_likelihood_{type}.pdf"), "pdf")
    fig.write_html(os.path.join(results_dir, f"average_log_likelihood_{type}.html"), "html")


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

    fig.update_yaxes(
        title_font=dict(size=title_font),
    )

    fig.update_xaxes(
        title_font=dict(size=title_font),
    )

    # Update layout to set legend font size
    fig.update_layout(legend=dict(font=dict(size=title_font)))


def add_traces(fig, values, local_scores, global_scores, local_color, global_color, row, col):
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


def plot_mauve_evaluation_metrics(results, model_names, results_dir):
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
        # subplot_titles=tuple(model_names),
        subplot_titles=(None,) * len(model_names),
    )

    for idx, model_name in enumerate(results.keys(), start=1):
        top_k_df = results[model_name]["top_k"]
        top_p_df = results[model_name]["top_p"]

        add_traces_with_ci(
            fig_top_k,
            top_k_values,
            top_k_df["mauve_local_mean"].tolist(),
            top_k_df["mauve_global_mean"].tolist(),
            top_k_df["mauve_local_ci"].tolist(),
            top_k_df["mauve_global_ci"].tolist(),
            local_color,
            global_color,
            1,
            idx,
            False,
        )
        add_traces_with_ci(
            fig_top_p,
            top_p_values,
            top_p_df["mauve_local_mean"].tolist(),
            top_p_df["mauve_global_mean"].tolist(),
            top_p_df["mauve_local_ci"].tolist(),
            top_p_df["mauve_global_ci"].tolist(),
            local_color,
            global_color,
            1,
            idx,
            True
        )

    max_top_k_score = max(
        score + 0.1 * score  # Adding 10% padding for the upper bound
        for model in results.values()
        for score in model["top_k"]["mauve_local_mean"].tolist() + model["top_k"]["mauve_global_mean"].tolist()
    )
    max_top_p_score = max(
        score + 0.1 * score  # Adding 10% padding for the upper bound
        for model in results.values()
        for score in model["top_p"]["mauve_local_mean"].tolist() + model["top_p"]["mauve_global_mean"].tolist()
    )

    update_fig(fig_top_k, max_top_k_score, "Top-k values", "MAUVE Score")
    update_fig(fig_top_p, max_top_p_score, "Top-π values", "MAUVE Score")

    fig_top_k.write_image(os.path.join(results_dir, "mauve_top_k.pdf"), "pdf")
    fig_top_k.write_html(os.path.join(results_dir, "mauve_top_k.html"), "html")

    fig_top_p.write_image(os.path.join(results_dir, "mauve_top_p.pdf"), "pdf")
    fig_top_p.write_html(os.path.join(results_dir, "mauve_top_p.html"), "html")


def plot_bleu_evaluation_metrics(results, model_names, results_dir):
    top_k_values = results[model_names[0]]["top_k"]["top_k"].tolist()
    top_p_values = results[model_names[0]]["top_p"]["top_p"].tolist()

    fig_top_k = make_subplots(
        rows=1,
        cols=4,
        shared_xaxes=True,
        horizontal_spacing=0.035,
        subplot_titles=tuple(model_names),
        # subplot_titles=(None,) * len(model_names),
    )
    fig_top_p = make_subplots(
        rows=1,
        cols=4,
        shared_xaxes=True,
        horizontal_spacing=0.035,
        # subplot_titles=tuple(model_names),
        subplot_titles=(None,) * len(model_names),
    )

    for idx, model_name in enumerate(results.keys(), start=1):
        top_k_df = results[model_name]["top_k"]
        top_p_df = results[model_name]["top_p"]

        add_traces_with_ci(
            fig_top_k,
            top_k_values,
            top_k_df["bleu_local_mean"].tolist(),
            top_k_df["bleu_global_mean"].tolist(),
            top_k_df["bleu_local_ci"].tolist(),
            top_k_df["bleu_global_ci"].tolist(),
            local_color,
            global_color,
            1,
            idx,
            False
        )
        add_traces_with_ci(
            fig_top_p,
            top_p_values,
            top_p_df["bleu_local_mean"].tolist(),
            top_p_df["bleu_global_mean"].tolist(),
            top_p_df["bleu_local_ci"].tolist(),
            top_p_df["bleu_global_ci"].tolist(),
            local_color,
            global_color,
            1,
            idx,
            True
        )

    max_top_k_score = max(
        score + 0.1 * score  # Adding 10% padding for the upper bound
        for model in results.values()
        for score in model["top_k"]["bleu_local_mean"].tolist() + model["top_k"]["bleu_global_mean"].tolist()
    )
    max_top_p_score = max(
        score + 0.1 * score  # Adding 10% padding for the upper bound
        for model in results.values()
        for score in model["top_p"]["bleu_local_mean"].tolist() + model["top_p"]["bleu_global_mean"].tolist()
    )

    update_fig(fig_top_k, max_top_k_score, "Top-k values", "Self-BLEU Score")
    update_fig(fig_top_p, max_top_p_score, "Top-π values", "Self-BLEU Score")

    fig_top_k.write_image(os.path.join(results_dir, "bleu_top_k.pdf"), "pdf")
    fig_top_k.write_html(os.path.join(results_dir, "bleu_top_k.html"), "html")

    fig_top_p.write_image(os.path.join(results_dir, "bleu_top_p.pdf"), "pdf")
    fig_top_p.write_html(os.path.join(results_dir, "bleu_top_p.html"), "html")


def replace_nan_ci(ci_list):
    """Replace NaN confidence intervals with [1.0, 1.0]."""
    return [[1.0, 1.0] if np.isnan(ci[0]) or np.isnan(ci[1]) else ci for ci in ci_list]


def add_traces_with_ci(
    fig, x_values, y_mean_local, y_mean_global, ci_local, ci_global, local_color, global_color, row, col, show_legend
):

    # Add mean traces
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_mean_local,
            mode="lines",
            line=dict(color=local_color),
            name="Local Decoding",
            showlegend=True if col == 1 and show_legend else False,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_mean_global,
            mode="lines",
            line=dict(color=global_color),
            name="Global Decoding",
            showlegend=True if col == 1 and show_legend else False,
        ),
        row=row,
        col=col,
    )

    # Add confidence interval traces
    y_local_upper = [ci[1] for ci in ci_local]
    y_local_lower = [ci[0] for ci in ci_local]
    y_global_upper = [ci[1] for ci in ci_global]
    y_global_lower = [ci[0] for ci in ci_global]

    # Fill for local mean
    fig.add_trace(
        go.Scatter(
            x=x_values + x_values[::-1],
            y=y_local_upper + y_local_lower[::-1],
            fill="toself",
            fillcolor="rgba(0, 103, 149, 0.2)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    # Fill for global mean
    fig.add_trace(
        go.Scatter(
            x=x_values + x_values[::-1],
            y=y_global_upper + y_global_lower[::-1],
            fill="toself",
            fillcolor="rgba(0, 155, 85, 0.2)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def plot_histograms(results, results_dir):
    for model_name, data in results.items():
        top_k_df = data["top_k"]
        # top_p_df = data["top_p"]

        # Create constants_products dict
        # constants_products = {
        #     "Top K": top_k_df["constants_products"].tolist(),
        #     "Top P": top_p_df["constants_products"].tolist(),
        # }
        # Use only top_k for now and make a dict top-k value to constants_products
        constants_products = {
            top_k: constants_products
            for top_k, constants_products in zip(top_k_df["top_k"].tolist(), top_k_df["constants_products"].tolist())
        }

        # Create subplots: each subplot for a different top_k value
        num_subplots = len(constants_products)
        fig = make_subplots(
            rows=1,
            cols=num_subplots,
            subplot_titles=[f"Top K = {top_k}" for top_k in constants_products.keys()],
        )

        # Add histograms to subplots
        for i, (top_k, constants) in enumerate(constants_products.items()):
            histogram = go.Histogram(
                x=constants,
                nbinsx=30,
                name=f"Top K = {top_k}",
                opacity=0.5,
                hoverinfo="all",
            )
            fig.add_trace(histogram, row=1, col=i + 1)

        # Update layout for the entire figure
        fig.update_layout(
            height=400,  # Adjust height as needed
            width=300 * num_subplots,  # Adjust width as needed
            # title_text=f"Histograms of Local Decoding Constants ({model_name})",
            font=dict(family="Times New Roman"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,  # Hide legend to avoid clutter
        )

        # Update x-axis properties for each subplot
        for i in range(num_subplots):
            fig.update_xaxes(
                title_text=r"Local decoding constant $\log \mathcal{c}_{\alpha}$",
                title_font=dict(family="Times New Roman", size=title_font),
                tickfont=dict(family="Times New Roman", size=tick_font),
                mirror=True,
                ticks="outside",
                showline=True,
                gridcolor="lightgrey",
                row=1,
                col=i + 1,
            )

        # Update y-axis properties for each subplot
        for i in range(num_subplots):
            if i == 0:
                title_text = "Frequency"
            else:
                title_text = ""
            fig.update_yaxes(
                title_text=title_text,
                title_font=dict(family="Times New Roman", size=title_font),
                tickfont=dict(family="Times New Roman", size=tick_font),
                mirror=True,
                ticks="outside",
                showline=True,
                gridcolor="lightgrey",
                row=1,
                col=i + 1,
            )

        # Save the figure as a PDF file
        fig.write_image(os.path.join(results_dir, f"histogram_{model_name}.pdf"), "pdf")
        fig.write_html(os.path.join(results_dir, f"histogram_{model_name}.html"), "html")


def generate_latex_table(results, model_names, results_dir, table_type="mauve"):
    assert table_type in ["mauve", "bleu"], "Invalid table type. Choose 'mauve' or 'bleu'."

    if table_type == "mauve":
        rounding_value = 3
    else:
        rounding_value = 4

    table_header = (
        r"""
    \begin{tabular}{l"""
        + "c" * (len(model_names) * 2)
        + r"""}
    \toprule
    & """
        + "".join([f"\multicolumn{{2}}{{c}}{{{model}}} & " for model in model_names]).rstrip(" & ")
        + r""" \\
    \cmidrule(lr){2-3}"""
        + "".join([f"\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(model_names))])
        + r"""
    & """
        + "".join(
            [
                r"$\mauvelocal$  & $\mauveglobal$ & " if table_type == "mauve" else r"$\bleulocal$  & $\bleuglobal$ & "
                for _ in model_names
            ]
        ).rstrip(" & ")
        + r""" \\\midrule
    """
    )

    table_body = ""

    # Retrieve top_k values and corresponding evaluation metrics for each model
    top_k_values = results[model_names[0]]["top_k"][
        "top_k"
    ].unique()  # Assuming top_k values are the same for all models

    for top_k in top_k_values:
        row = f"$\\alphabetkeeptopkwithval{{{top_k}}}$    "
        for model in model_names:
            model_data = results[model]["top_k"].loc[results[model]["top_k"]["top_k"] == top_k]
            if not model_data.empty:
                local_mean = round(model_data[f"{table_type}_local_mean"].values[0], rounding_value)
                local_ci_low, local_ci_high = model_data[f"{table_type}_local_ci"].values[0]
                local_ci = round(local_ci_high - local_ci_low, rounding_value)

                global_mean = round(model_data[f"{table_type}_global_mean"].values[0], rounding_value)
                global_ci_low, global_ci_high = model_data[f"{table_type}_global_ci"].values[0]
                global_ci = round(global_ci_high - global_ci_low, rounding_value)

                row += f" & {local_mean} ± {local_ci} & {global_mean} ± {global_ci}"
            else:
                row += " & - & -"
        row += r" \\ \midrule" + "\n"
        table_body += row

    # Add top_p values (similar to top_k)
    top_p_values = results[model_names[0]]["top_p"]["top_p"].unique()

    for top_p in top_p_values:
        row = f"$\\alphabetkeeptoppwithval{{{top_p}}}$    "
        for model in model_names:
            model_data = results[model]["top_p"].loc[results[model]["top_p"]["top_p"] == top_p]
            if not model_data.empty:
                local_mean = round(model_data[f"{table_type}_local_mean"].values[0], rounding_value)
                local_ci_low, local_ci_high = model_data[f"{table_type}_local_ci"].values[0]
                local_ci = round(local_ci_high - local_ci_low, rounding_value)

                global_mean = round(model_data[f"{table_type}_global_mean"].values[0], rounding_value)
                global_ci_low, global_ci_high = model_data[f"{table_type}_global_ci"].values[0]
                global_ci = round(global_ci_high - global_ci_low, rounding_value)

                row += f" & {local_mean} ± {local_ci} & {global_mean} ± {global_ci}"
            else:
                row += " & - & -"
        row += r" \\ \midrule" + "\n"
        table_body += row

    table_footer = r"""\bottomrule
    \end{tabular}
    """

    # Combine header, body, and footer
    table_content = table_header + table_body + table_footer

    # Save the LaTeX table to a file
    output_file = os.path.join(results_dir, f"{table_type}_latex_table.tex")
    with open(output_file, "w") as f:
        f.write(table_content)
