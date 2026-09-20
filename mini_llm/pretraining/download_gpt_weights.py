# Download OpenAI's original GPT-2 weights and load them into plain numpy arrays.
#
# OpenAI never released these as PyTorch state dicts -- they are TensorFlow v1
# checkpoints sitting in a public Azure blob container, which is why tensorflow
# is a dependency here even though the model itself is pure PyTorch. We read the
# checkpoint once, convert every variable to numpy, and hand back a nested dict
# that mirrors the GPT-2 architecture so the weights can be copied into GPTModel.

import json
import os
import urllib.request
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models"

# the four sizes OpenAI published; the name doubles as the directory name in the
# blob container, so it has to be spelled exactly like this
MODEL_SIZES = ("124M", "355M", "774M", "1558M")

# every checkpoint directory ships this same set of files. hparams.json gives us
# the architecture settings, the three model.ckpt.* files are the weights, and
# the encoder/vocab files are the BPE tokenizer (we use tiktoken instead, but
# they're small and keep the directory self-contained)
FILENAMES = (
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe",
)

DEFAULT_MODELS_DIR = PROJECT_ROOT / "gpt2"


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    """Stream `url` to `destination`, skipping the download if it's already complete.

    The 1558M checkpoint is ~6 GB, so a resumed session should not re-fetch what
    is already on disk. We compare against Content-Length rather than merely
    checking existence, because a run interrupted mid-download leaves a
    truncated file behind that would otherwise look valid.
    """
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get("Content-Length", 0))

        if destination.exists() and destination.stat().st_size == total_size:
            print(f"{destination.name} already up to date, skipping")
            return

        # write to a temp name first so an interrupted run can never leave a
        # half-written file that the size check above would accept
        partial = destination.with_suffix(destination.suffix + ".part")
        progress = tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc=destination.name,
        )
        with open(partial, "wb") as f, progress:
            while chunk := response.read(chunk_size):
                f.write(chunk)
                progress.update(len(chunk))

    os.replace(partial, destination)


def download_gpt2_files(model_size: str, models_dir: Path) -> Path:
    """Fetch all checkpoint files for `model_size` and return their directory."""
    if model_size not in MODEL_SIZES:
        raise ValueError(
            f"model_size must be one of {MODEL_SIZES}, got {model_size!r}"
        )

    model_dir = Path(models_dir) / model_size
    model_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILENAMES:
        download_file(f"{BASE_URL}/{model_size}/{filename}", model_dir / filename)

    return model_dir


def load_gpt2_params_from_tf_ckpt(ckpt_path: str, settings: dict) -> dict:
    """Convert the TF checkpoint into a nested dict of numpy arrays.

    TF stores variables under flat slash-separated names like
    `model/h0/attn/c_attn/w`. We rebuild the nesting those names imply: the per
    layer `h<N>` variables go into `params["blocks"][N]`, everything else
    (`wte`, `wpe`, the final `ln_f`) stays at the top level.
    """
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        # squeeze because TF keeps the conv1d weights as (1, in, out); the
        # matching PyTorch Linear layers want plain 2-D matrices
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # drop the leading "model/" scope, which every variable shares
        name_parts = name.split("/")[1:]

        target_dict = params
        if name_parts[0].startswith("h"):
            layer_number = int(name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # walk/create the intermediate scopes, e.g. "attn" -> "c_attn"
        for key in name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        target_dict[name_parts[-1]] = variable_array

    return params


def download_and_load_gpt2(
    model_size: str = "124M", models_dir: Path | str = DEFAULT_MODELS_DIR
) -> tuple[dict, dict]:
    """Download `model_size` if needed and return `(settings, params)`.

    `settings` is OpenAI's hparams.json (n_embd, n_head, n_layer, n_ctx,
    n_vocab); `params` is the nested numpy weight dict.
    """
    model_dir = download_gpt2_files(model_size, Path(models_dir))

    settings = json.loads((model_dir / "hparams.json").read_text(encoding="utf-8"))
    # tf.train wants the checkpoint prefix, not an actual file on disk
    params = load_gpt2_params_from_tf_ckpt(str(model_dir / "model.ckpt"), settings)

    return settings, params


@click.command()
@click.option(
    "--model-size",
    default="124M",
    type=click.Choice(MODEL_SIZES),
    show_default=True,
    help="which GPT-2 checkpoint to download",
)
@click.option(
    "--models-dir",
    default=DEFAULT_MODELS_DIR,
    type=click.Path(file_okay=False, path_type=Path),
    show_default=True,
    help="where to store the checkpoints",
)
def main(model_size: str, models_dir: Path) -> None:
    """Download the original OpenAI GPT-2 weights."""
    settings, params = download_and_load_gpt2(model_size, models_dir)

    click.echo(f"\nSettings: {settings}")
    click.echo(f"Parameter dictionary keys: {list(params.keys())}")
    click.echo(f"Token embedding weight tensor shape: {params['wte'].shape}")
    click.echo(f"Number of transformer blocks: {len(params['blocks'])}")


if __name__ == "__main__":
    main()
