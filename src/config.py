"""
Configuration and settings for the diarization pipeline.
"""

import argparse
import os
from pathlib import Path

# Supported audio extensions
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}

# Default settings
DEFAULT_MIN_DURATION = 0.5      # seconds - skip clips shorter than this
DEFAULT_OUTPUT_FORMAT = "wav"
DEFAULT_MERGE_GAP = 0.3         # seconds - max gap to merge same-speaker segments
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"


def get_hf_token() -> str:
    """Load the Hugging Face token from environment (set via .env)."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError(
            "Hugging Face token not found.\n"
            "  Set HF_TOKEN in your .env file or export it as an env variable.\n"
            "  Get a token at: https://hf.co/settings/tokens\n"
            "  Accept model terms at: https://hf.co/pyannote/speaker-diarization-3.1"
        )
    return token


def parse_args():
    """Parse command-line arguments for the batch pipeline."""
    parser = argparse.ArgumentParser(
        description="Batch speaker diarization pipeline. "
                    "Processes all audio files in the input folder.",
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing audio files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--token", "-t",
        default=None,
        help="Hugging Face API token (overrides .env / HF_TOKEN)",
    )
    parser.add_argument(
        "--num-speakers", "-n",
        type=int,
        default=None,
        help="Known number of speakers (leave blank to auto-detect)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        help=f"Minimum clip duration in seconds (default: {DEFAULT_MIN_DURATION})",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["wav", "mp3", "flac"],
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output audio format (default: {DEFAULT_OUTPUT_FORMAT})",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable merging of consecutive same-speaker segments",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=DEFAULT_MERGE_GAP,
        help=f"Max gap (seconds) to merge same-speaker segments (default: {DEFAULT_MERGE_GAP})",
    )
    return parser.parse_args()


def discover_audio_files(input_dir: str) -> list[Path]:
    """Find all supported audio files in the input directory."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = []
    for f in sorted(input_path.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(f)

    return files
