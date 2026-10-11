"""
Root CLI entrypoint.

Data-collection pipeline:
    python main.py list-models
    python main.py generate ...
    python main.py score ...

"""

from pathlib import Path
import sys
from rich.table import Table
import typer
from dotenv import load_dotenv
from rich import print
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# Support `python main.py ...` from the repo root without requiring installation.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_registry import list_supported_model_keys  # noqa: E402

app = typer.Typer(add_completion=False)

load_dotenv()

SCORER_DEFAULTS = {
    "armorm": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
    "bart":   "facebook/bart-large",
}


@app.command("list-models")
def list_models() -> None:
    """Print all supported model names (model keys)."""
    print("[bold]Supported models:[/bold]")
    for name in list_supported_model_keys():
        print(f" - {name}")


@app.command()
def generate(
    data_path: str = typer.Option(
        ..., "--data-path", help="Input JSONL with {'id','prompt'} per line."
    ),
    out_path: str = typer.Option("out.jsonl", "--out-path", help="Output JSONL file."),
    model_name: str = typer.Option(
        ..., "--model-name", help="Model key (e.g., gpt-4o, phi-3-mini)."
    ),
    num_samples: int = typer.Option(
        20, "--num-samples", help="How many responses per prompt (usually 20)."
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", help="Sampling temperature."
    ),
    top_p: float = typer.Option(1.0, "--top-p", help="Nucleus sampling (top_p)."),
    max_new_tokens: int = typer.Option(
        256, "--max-new-tokens", help="Max tokens per response."
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only process first K prompts (debug)."),
    start_index: int = typer.Option(
    0,
    "--start-index",
    help="Start processing from this prompt index."
),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Skip prompts already in output file."
    ),
) -> None:
    """Generate multiple responses per prompt and save to JSONL."""
    from sampling_pipeline import (
        generate_and_save_responses_for_dataset,
    )

    print(f"[bold]Model key:[/bold] {model_name}")
    print(f"[bold]Input:[/bold] {data_path}")
    print(f"[bold]Output:[/bold] {out_path}")

    generate_and_save_responses_for_dataset(
        model_key=model_name,
        input_prompts_jsonl_path=data_path,
        output_responses_jsonl_path=out_path,
        number_of_samples_per_prompt=num_samples,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        max_number_of_prompts=limit,
        resume_if_output_exists=resume,
        start_index=start_index
    )

    print("[green]Done.[/green]")


@app.command()
def score(
    data_path: str = typer.Option(
        ..., "--data-path", help="Input JSONL with responses (output of generate step)."
    ),
    out_path: str = typer.Option(
        "scored.jsonl", "--out-path", help="Output scored JSONL file."
    ),
    scorer: str = typer.Option(
        "armorm", "--scorer", help="Scorer to use: 'armorm' or 'bart'."
    ),
    model_name: str = typer.Option(
        None, "--model-name", help="HuggingFace model name for the scorer. Defaults to the scorer's standard model."
    ),
    limit: Optional[int] = typer.Option(
    None, "--limit", help="Only process first K prompts (debug)."),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Skip prompts already in output file."
    ),
) -> None:
    """Score model responses using a chosen scorer (armorm or bart)."""
    scorer = scorer.lower()
    if scorer not in SCORER_DEFAULTS:
        print(f"[red]Unknown scorer '{scorer}'. Choose from: {', '.join(SCORER_DEFAULTS)}[/red]")
        raise typer.Exit(code=1)

    resolved_model = model_name or SCORER_DEFAULTS[scorer]

    print(f"[bold]Input:[/bold]   {data_path}")
    print(f"[bold]Output:[/bold]  {out_path}")
    print(f"[bold]Scorer:[/bold]  {scorer}")
    print(f"[bold]Model:[/bold]   {resolved_model}")

    if scorer == "armorm":
        from armorm_scoring_pipeline import score_and_save_responses
    elif scorer == "bart":
        from bart_scoring_pipeline import score_and_save_responses

    score_and_save_responses(
        input_responses_jsonl_path=data_path,
        output_scored_jsonl_path=out_path,
        model_name=resolved_model,
        max_number_of_prompts=limit,
        resume_if_output_exists=resume,
    )

    print("[green]Done.[/green]")

@app.command("quick-eval")
def quick_eval(
    prompt: str = typer.Option(..., "--prompt", help="Prompt to evaluate."),
    model_name: str = typer.Option(
        ..., "--model-name", help="Generator model key."
    ),
    scorer: str = typer.Option(
        "armorm", "--scorer", help="armorm or bart"
    ),
    num_samples: int = typer.Option(
        1, "--num-samples"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature"
    ),
):
    """
    Generate responses and immediately score them.
    """

    from sampling_pipeline import generate_responses_for_prompt

    scorer = scorer.lower()

    if scorer == "armorm":
        from armorm_scoring_pipeline import score_single_response
    elif scorer == "bart":
        from bart_scoring_pipeline import score_single_response
    else:
        print(f"[red]Unknown scorer: {scorer}[/red]")
        raise typer.Exit(code=1)

    responses = generate_responses_for_prompt(
        prompt_text=prompt,
        model_key=model_name,
        number_of_samples=num_samples,
        temperature=temperature,
        top_p = 0.9,
        max_new_tokens = 256
    )

    print("\n[bold]Results[/bold]\n")

    for i, response in enumerate(responses["responses"]):
        text = response["response_text"]

        score = score_single_response(prompt = prompt, response = text)

        print(f"[bold]Sample {i}[/bold]")
        print(text)
        print()

        if scorer == "armorm":
            print(f"ArmoRM score: {score['armorm_score']}")
        else:
            print(f"BARTScore: {score['bart_score']}")

        print("-" * 80)

    ARMORM_DIMENSIONS = [
        "helpsteer-helpfulness",
        "helpsteer-correctness",
        "helpsteer-coherence",
        "helpsteer-complexity",
        "helpsteer-verbosity",
        "ultrafeedback-overall-score",
        "ultrafeedback-instruction-following",
        "ultrafeedback-truthfulness",
        "ultrafeedback-honesty",
        "ultrafeedback-helpfulness",
        "beavertails-is-safe",
        "code-complexity",
        "code-style",
        "code-explanation",
        "code-instruction-following",
        "code-readability",
        "prometheus-score",
        "argilla-overall-quality",
        "argilla-judge-lm",]

    
    table = Table(title="ArmoRM Dimension Analysis")

    table.add_column("Dimension")
    table.add_column("Reward")
    table.add_column("Gate")
    table.add_column("Contribution")

    for dim, reward, gate in zip(
        ARMORM_DIMENSIONS,
        score["armorm_rewards"],
        score["armorm_gating"],
    ):
        contribution = reward * gate

        table.add_row(
            dim,
            f"{reward:.4f}",
            f"{gate:.4f}",
            f"{contribution:.4f}",
        )
    print(responses)
    print(table)
    print(f"\nFinal ArmoRM Score: {score['armorm_score']:.4f}")



if __name__ == "__main__":
    app()