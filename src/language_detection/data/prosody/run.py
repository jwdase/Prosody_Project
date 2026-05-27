"""Batch driver for prosody-only synthesis.

Python port of ``src/matlab/run_many_high_performance.m``. Reads audio file
lists (one path per line, ``#`` comments allowed), resolves the per-language
output directory from a ``config.json`` mapping, and synthesises a prosody-only
.wav for each input in parallel.

Usage:
    python -m language_detection.data.prosody.run data/it.txt
    python -m language_detection.data.prosody.run data/            # all *.txt
"""

import argparse
import json
import time
from pathlib import Path

from joblib import Parallel, delayed

from language_detection.data.prosody.synth import synth_only_prosody


def read_audio_list(list_path, output_root):
    """Return (input_path, output_path) pairs from a newline-delimited list.

    Relative input paths are resolved against the list file's directory; each
    output keeps the input's base name with a .wav extension.
    """
    list_path = Path(list_path)
    list_dir = list_path.parent
    output_root = Path(output_root)

    pairs = []
    for line in list_path.read_text().splitlines():
        audio = line.strip()
        if not audio or audio.startswith("#"):
            continue
        audio_path = Path(audio)
        if not audio_path.is_absolute():
            audio_path = list_dir / audio_path
        pairs.append((audio_path, output_root / (audio_path.stem + ".wav")))
    return pairs


def get_output_dir(cfg, lang):
    """Look up (and create) the output directory for a language."""
    if lang not in cfg:
        raise KeyError(
            f"Language '{lang}' not found in config. Available: {', '.join(cfg)}"
        )
    out = Path(cfg[lang])
    out.mkdir(parents=True, exist_ok=True)
    return out


def _process_one(input_path, output_path):
    """Synthesise a single file; skip if the output already exists."""
    output_path = Path(output_path)
    if output_path.exists():
        return True, None
    if not Path(input_path).exists():
        return False, f"Input audio not found: {input_path}"
    try:
        synth_only_prosody(input_path, output_path)
        return True, None
    except Exception as exc:  # report per-file failures, keep the batch going
        return False, str(exc)


def collect_pairs(input_path, cfg):
    """Build the full (input, output) work list from a file or a directory."""
    input_path = Path(input_path)
    pairs = []
    if input_path.is_dir():
        lists = sorted(input_path.rglob("*.txt"))
        if not lists:
            raise FileNotFoundError(f"No .txt files found under: {input_path}")
        for list_path in lists:
            pairs += read_audio_list(list_path, get_output_dir(cfg, list_path.stem))
    else:
        pairs = read_audio_list(input_path, get_output_dir(cfg, input_path.stem))
    return pairs


def run_many(input_path, config_path="config.json", n_jobs=-1):
    """Process every audio file referenced by ``input_path``.

    Args:
        input_path: a list file (``<lang>.txt``) or a directory of list files.
        config_path: JSON mapping of language -> output directory.
        n_jobs: parallel workers (-1 uses all cores).
    """
    cfg = json.loads(Path(config_path).read_text())
    pairs = collect_pairs(input_path, cfg)

    total = len(pairs)
    print(f"Total audio files to process: {total}")
    if total == 0:
        return

    start = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_one)(inp, out) for inp, out in pairs
    )
    elapsed = time.time() - start

    ok = sum(1 for success, _ in results if success)
    failed = total - ok
    print("=== FINAL SUMMARY ===")
    print(f"Successful: {ok}  Failed: {failed}")
    print(f"Total time: {elapsed:.1f}s  ({elapsed / total:.2f}s per file)")

    if failed:
        print("Errors (up to 10 shown):")
        shown = 0
        for (inp, _), (success, msg) in zip(pairs, results):
            if not success:
                print(f"  - {inp}: {msg}")
                shown += 1
                if shown >= 10:
                    break


def main():
    parser = argparse.ArgumentParser(description="Prosody-only batch synthesis")
    parser.add_argument("input_path", help="list file (<lang>.txt) or directory")
    parser.add_argument("--config", default="config.json", help="language->dir map")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel workers")
    args = parser.parse_args()
    run_many(args.input_path, args.config, args.jobs)


if __name__ == "__main__":
    main()
