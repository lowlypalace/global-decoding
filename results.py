import os
import json
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import logging
import plotly.io as pio

# To avoid bug in graphs
pio.kaleido.scope.mathjax = None

def print_bleu_scores(results):
    for result in results:
        print(f"Top-k: {result['top_k']}")
        print(f"Global Self-BLEU: {result['global_bleu']}")
        print(f"Local Self-BLEU: {result['local_bleu']}")
        print("-" * 30)

def filter_padding_tokens(sequence):
    """Helper function to filter out padding tokens (tokens with value 0)."""
    return [token for token in sequence if token != 0]


def get_results(model_name):
    base_dir = os.path.join("output", model_name)
    results = []

    for sub_dir in os.listdir(base_dir):
        sequences_dir = os.path.join(base_dir, sub_dir, "sequences")
        mcmc_dir = os.path.join(base_dir, sub_dir, "mcmc")
        eval_dir = os.path.join(base_dir, sub_dir, "eval")

        # logging.info("base_dir")

        if os.path.exists(sequences_dir) and os.path.exists(mcmc_dir):
            try:
                # Load metadata
                with open(os.path.join(base_dir, sub_dir, "metadata.json"), "r") as f:
                    metadata = json.load(f)
                top_k = metadata.get("top_k")
                top_p = metadata.get("top_p")

                # TODO: handle missing results by skipping (if metadata exists but some files not)

                # when top_k and top_p are equal to None
                if top_k is None and top_p is None:
                    top_k = 50432
                    top_p = 1.0

                ##################
                # MAUVE
                ##################

                with open(
                    os.path.join(eval_dir, "mauve_results_global.json"), "r"
                ) as f:
                    global_result = json.load(f)
                global_mauve = global_result.get("mauve")

                # Load local MAUVE result
                with open(os.path.join(eval_dir, "mauve_results_local.json"), "r") as f:
                    local_result = json.load(f)
                local_mauve = local_result.get("mauve")

                ##################
                # BLEU
                ##################

                with open(os.path.join(eval_dir, "self_bleu_results.json"), "r") as f:
                    bleu = json.load(f)
                local_bleu = bleu.get("local_self_bleu")
                global_bleu = bleu.get("global_self_bleu")

                ##################
                # Sequences Lengths
                ##################

                with open(os.path.join(sequences_dir, "sequences_ids.json"), "r") as f:
                    sequences_data = json.load(f)

                with open(
                    os.path.join(mcmc_dir, "sampled_sequences_ids.json"), "r"
                ) as f:
                    mcmc_data = json.load(f)

                # Sample first 200 random sequences
                random_sequences = random.sample(
                    sequences_data, min(len(sequences_data), 200)
                )

                # Filter padding tokens and compute average lengths
                filtered_sequences = [
                    filter_padding_tokens(seq) for seq in random_sequences
                ]
                filtered_mcmc = [filter_padding_tokens(seq) for seq in mcmc_data]

                avg_length_sequences = (
                    sum(len(seq) for seq in filtered_sequences)
                    / len(filtered_sequences)
                    if filtered_sequences
                    else 0
                )
                avg_length_mcmc = (
                    sum(len(seq) for seq in filtered_mcmc) / len(filtered_mcmc)
                    if filtered_mcmc
                    else 0
                )

                ##################
                # Example Sequences
                ##################

                with open(
                    os.path.join(sequences_dir, "sequences_decoded.json"), "r"
                ) as f:
                    sequences_decoded = json.load(f)
                sequence_decoded = random.sample(sequences_decoded, 1)[0][:100]
                sequence_decoded = sequence_decoded.replace('\n', '\\n')

                with open(
                    os.path.join(mcmc_dir, "sampled_sequences_decoded.json"), "r"
                ) as f:
                    sampled_sequences_decoded = json.load(f)
                sequence_decoded_sampled = random.sample(sampled_sequences_decoded, 1)[
                    0
                ][:100]
                sequence_decoded_sampled = sequence_decoded_sampled.replace('\n', '\\n')

                ###################
                # Decoding constants:
                # TODO
                # Compute product of local normalization constants
                ###################

                ###################
                # Log likelihood
                ###################
                with open(
                    os.path.join(mcmc_dir, "sampled_target_logprobs.json"), "r"
                ) as f:
                    sampled_target_logprobs = json.load(f)

                average_log_likelihood = sum(sampled_target_logprobs) / len(sampled_target_logprobs)

                results.append(
                    {
                        "sub_dir": sub_dir,
                        "top_k": top_k,
                        "top_p": top_p,
                        "mauve_local": local_mauve,
                        "mauve_global": global_mauve,
                        "bleu_local": local_bleu,
                        "global_bleu": global_bleu,
                        "average_log_likelihood": average_log_likelihood,
                        "avg_length_local": avg_length_sequences,
                        "avg_length_global": avg_length_mcmc,
                        "sequence_local": sequence_decoded,
                        "sequence_global": sequence_decoded_sampled,

                    }
                )
            except Exception as e:
                print(f"Error processing {sub_dir}: {e}")

    results_df = pd.DataFrame(results)

    # Sort
    top_k_df = results_df.dropna(subset=["top_k"]).sort_values(by="top_k")
    top_p_df = results_df.dropna(subset=["top_p"]).sort_values(by="top_p")

    return top_k_df, top_p_df


def plot_sequences_lengths(top_k_df, top_p_df, results_dir):

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
    fig.update_xaxes(mirror=True, ticks="outside", showline=True, gridcolor="lightgrey")

    # Update y-axes properties
    fig.update_yaxes(mirror=True, ticks="outside", showline=True, gridcolor="lightgrey")

    # Update layout background color
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # Update layout
    fig.update_layout(
        height=360,
        width=1400,
        font=dict(family="Times New Roman", size=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
    )

    # Save the figure as an EPS file
    fig.write_image(os.path.join(results_dir, "average_lengths.pdf"), format="pdf")

def plot_average_log_likelihood(top_k_70, top_p_70, top_k_410, top_p_410, top_k_14, top_p_14, results_dir):
    # Create subplots for top-k and top-p
    fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.07)

    # Define colors
    log_likelihood_color_70 = "#92268F"
    log_likelihood_color_410 = "#FBB982"
    log_likelihood_color_14 = "#41B0E4"

    # Add line traces for top-k
    fig.add_trace(
        go.Scatter(
            x=top_k_70["top_k"],
            y=top_k_70["average_log_likelihood"],
            mode="lines+markers",
            name="pythia-70m",
            marker_color=log_likelihood_color_70,

        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=top_k_410["top_k"],
            y=top_k_410["average_log_likelihood"],
            mode="lines+markers",
            name="pythia-410m",
            marker_color=log_likelihood_color_410,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=top_k_14["top_k"],
            y=top_k_14["average_log_likelihood"],
            mode="lines+markers",
            name="pythia-1.4b",
            marker_color=log_likelihood_color_14,
        ),
        row=1,
        col=1,
    )

    # Add line traces for top-p
    fig.add_trace(
        go.Scatter(
            x=top_p_70["top_p"],
            y=top_p_70["average_log_likelihood"],
            mode="lines+markers",
            name="Average Log Likelihood",
            marker_color=log_likelihood_color_70,
            showlegend=False,
        ),
        row = 1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=top_p_410["top_p"],
            y=top_p_410["average_log_likelihood"],
            mode="lines+markers",
            name="Average Log Likelihood",
            marker_color=log_likelihood_color_410,
            showlegend=False,
        ),
        row = 1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=top_p_14["top_p"],
            y=top_p_14["average_log_likelihood"],
            mode="lines+markers",
            name="Average Log Likelihood",
            marker_color=log_likelihood_color_14,
            showlegend=False,
        ),
        row = 1,
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
        font=dict(family="Times New Roman", size=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
    )

    # Save the figure as a PDF file
    fig.write_image(os.path.join(results_dir, "average_log_likelihood.pdf"), format="pdf")


def main():
    results_dir = "results"  # TODO: Accept as arg
    model_name = "pythia-1.4b"  # TODO: Accept as arg

    # Create a directory to save the output files
    os.makedirs(results_dir, exist_ok=True)

    # print_bleu_scores(results)

    top_k_df, top_p_df = get_results(model_name)

    top_k_df.to_csv(os.path.join(results_dir, f"top_k_{model_name}.csv"), sep="\t")
    top_p_df.to_csv(os.path.join(results_dir, f"top_p_{model_name}.csv"), sep="\t")

    # plot_sequences_lengths(top_k_df, top_p_df, results_dir)

    # top_k_70, top_p_70 = get_results('pythia-70m')
    # top_k_410, top_p_410 = get_results('pythia-410m')
    # top_k_14, top_p_14 = get_results('pythia-1.4b')

    # # Plot average log likelihood
    # plot_average_log_likelihood(top_k_70, top_p_70, top_k_410, top_p_410, top_k_14, top_p_14, results_dir)

    # TODO: plot MAUVE / BLEU for each model


if __name__ == "__main__":
    main()
