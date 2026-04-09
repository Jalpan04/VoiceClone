"""
Batch Speaker Diarization Pipeline
===================================
Drop audio files into the input/ folder and run:

    python run.py

Each audio file gets its own subfolder in output/ with:
  - Per-speaker subfolders containing labeled clips
  - A diarization JSON file
  - An RTTM file

All clip filenames include the source audio name for traceability.

Usage:
    python run.py                              # uses input/ and output/ defaults
    python run.py -i my_audios -o results      # custom dirs
    python run.py --num-speakers 2 --format mp3
"""

import os
import sys
import time
from pathlib import Path

# Load .env for HF_TOKEN before anything else
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from src.config import parse_args, get_hf_token, discover_audio_files
from src.pipeline import load_pipeline
from src.diarizer import diarize, parse_segments
from src.splitter import load_audio, split_audio
from src.exporter import save_json, save_rttm, print_summary
from src.utils import convert_to_wav, cleanup_temp


def process_one_file(
    audio_path: Path,
    pipeline,
    output_root: str,
    num_speakers: int | None,
    min_duration: float,
    output_format: str,
    merge: bool,
    merge_gap: float,
):
    """Process a single audio file end-to-end."""
    source_name = audio_path.stem
    file_output_dir = os.path.join(output_root, source_name)
    os.makedirs(file_output_dir, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"  Processing: {audio_path.name}")
    print(f"  Output to : {file_output_dir}")
    print(f"{'#' * 60}")

    # 1. Convert to WAV if needed (torchaudio is more reliable with WAV)
    wav_path = convert_to_wav(str(audio_path), file_output_dir)

    try:
        # 2. Diarize
        raw_output, annotation = diarize(pipeline, wav_path, num_speakers)

        # 3. Parse segments
        segments = parse_segments(annotation)
        print(f"[INFO] Found {len(segments)} speech segments")

        # 4. Export metadata
        save_rttm(annotation, file_output_dir, source_name)
        save_json(segments, file_output_dir, source_name)

        # 5. Split audio into clips (load original file for best quality)
        audio = load_audio(str(audio_path))
        output_files = split_audio(
            audio=audio,
            segments=segments,
            output_dir=file_output_dir,
            source_name=source_name,
            min_duration=min_duration,
            output_format=output_format,
            merge_consecutive=merge,
            merge_gap_threshold=merge_gap,
        )

        # 6. Summary
        print_summary(segments, output_files, source_name)
        return True

    finally:
        # 7. Clean up temp WAV
        cleanup_temp(wav_path, str(audio_path))


def main():
    args = parse_args()

    # Resolve token
    if args.token:
        os.environ["HF_TOKEN"] = args.token
    try:
        hf_token = get_hf_token()
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Discover audio files
    try:
        audio_files = discover_audio_files(args.input)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(f"        Create the '{args.input}' folder and put your audio files in it.")
        sys.exit(1)

    if not audio_files:
        print(f"[ERROR] No audio files found in: {args.input}")
        print(f"        Supported formats: mp3, wav, flac, ogg, m4a, aac, wma, opus")
        sys.exit(1)

    print("=" * 60)
    print("  BATCH SPEAKER DIARIZATION PIPELINE")
    print("=" * 60)
    print(f"  Input dir   : {args.input}")
    print(f"  Output dir  : {args.output}")
    print(f"  Files found : {len(audio_files)}")
    print(f"  Format      : {args.format}")
    print(f"  Min duration: {args.min_duration}s")
    print(f"  Speakers    : {'auto' if args.num_speakers is None else args.num_speakers}")
    print("=" * 60)

    for i, f in enumerate(audio_files, 1):
        print(f"  {i}. {f.name}")
    print()

    # Load pipeline ONCE (reused for all files)
    pipeline = load_pipeline(hf_token)

    # Process each file
    succeeded = 0
    failed = []
    t0 = time.time()

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] ", end="")
        try:
            process_one_file(
                audio_path=audio_file,
                pipeline=pipeline,
                output_root=args.output,
                num_speakers=args.num_speakers,
                min_duration=args.min_duration,
                output_format=args.format,
                merge=not args.no_merge,
                merge_gap=args.merge_gap,
            )
            succeeded += 1
        except Exception as e:
            print(f"[FAILED] {audio_file.name}: {e}")
            failed.append((audio_file.name, str(e)))

    elapsed = time.time() - t0

    # Final report
    print(f"\n{'=' * 60}")
    print(f"  BATCH COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Processed : {succeeded}/{len(audio_files)} files")
    print(f"  Time      : {elapsed:.1f}s")
    if failed:
        print(f"  Failed    :")
        for name, err in failed:
            print(f"    - {name}: {err}")
    print(f"  Results in: {args.output}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
