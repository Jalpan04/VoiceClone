import os
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

OUTPUT_DIR = "output"
LOG_FILE = "annotations.json"
AUTOAOT_FILE = "autoaot.json"

# Mount static folders
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.mount("/static", StaticFiles(directory="static"), name="static")


class AnnotationBody(BaseModel):
    path: str
    label: str


def load_annotations():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_annotations(data):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_autoaot():
    if os.path.exists(AUTOAOT_FILE):
        with open(AUTOAOT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_autoaot(data):
    with open(AUTOAOT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_speaker_folders():
    """
    Finds all speaker folders like output/source_name/SPEAKER_XX.
    Returns a sorted list of relative folder paths.
    """
    speaker_folders = []
    if not os.path.exists(OUTPUT_DIR):
        return speaker_folders

    for source in os.listdir(OUTPUT_DIR):
        source_path = os.path.join(OUTPUT_DIR, source)
        if os.path.isdir(source_path):
            for speaker in os.listdir(source_path):
                speaker_path = os.path.join(source_path, speaker)
                if os.path.isdir(speaker_path) and "SPEAKER" in speaker:
                    speaker_folders.append(speaker_path.replace("\\", "/"))

    return sorted(speaker_folders)


def get_top_clips(speaker_folder_path):
    """
    Return the top 2 longest .wav clips in a speaker folder.
    """
    exts = {".wav", ".mp3", ".m4a", ".flac"}
    clips = []

    if not os.path.exists(speaker_folder_path):
        return []

    for f in os.listdir(speaker_folder_path):
        if Path(f).suffix.lower() in exts and not f.endswith("_temp.wav"):
            filepath = os.path.join(speaker_folder_path, f).replace("\\", "/")
            try:
                time_range = f.rsplit("_", 1)[-1].rsplit(".", 1)[0]
                start_str, end_str = time_range.split("-")
                duration = float(end_str.rstrip("s")) - float(start_str.rstrip("s"))
            except Exception:
                duration = 0
            clips.append((duration, filepath))

    clips.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in clips[:2]]


def build_queue(annotations, autoaot, all_folders):
    """
    Build the review queue with priority ordering:
      1. 'maybe' items from autoaot.json (not yet in annotations as mine/not_mine)
      2. 'not_mine' items from annotations.json for re-review
    Items already labeled 'mine' are excluded entirely.
    """
    mine_set = {k for k, v in annotations.items() if v == "mine"}

    # Phase 1: unsure items from autoaot that are not yet resolved as mine
    maybe_queue = [
        (k, "maybe", v.get("similarity"))
        for k, v in autoaot.items()
        if k not in mine_set and k not in annotations
    ]
    # Sort by similarity descending (highest first — most likely mine)
    maybe_queue.sort(key=lambda x: x[2] if x[2] else 0, reverse=True)

    # Phase 2: not_mine items needing re-review (excluding mine set)
    not_mine_queue = [
        (k, "not_mine", None)
        for k, v in annotations.items()
        if v == "not_mine"
    ]
    not_mine_queue.sort(key=lambda x: x[0])

    return maybe_queue + not_mine_queue


@app.get("/")
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/next")
def get_next():
    annotations = load_annotations()
    autoaot = load_autoaot()
    all_folders = get_speaker_folders()

    queue = build_queue(annotations, autoaot, all_folders)

    total_mine = sum(1 for v in annotations.values() if v == "mine")
    total_maybe = sum(1 for k in autoaot if k not in annotations)
    total_not_mine = sum(1 for v in annotations.values() if v == "not_mine")

    if not queue:
        return {
            "done": True,
            "remaining": 0,
            "total_mine": total_mine,
            "total_maybe": total_maybe,
            "total_not_mine": total_not_mine,
        }

    folder_path, mode, similarity = queue[0]
    top_clips = get_top_clips(folder_path)

    return {
        "done": False,
        "remaining": len(queue),
        "path": folder_path,
        "top_clips": top_clips,
        "mode": mode,           # "maybe" or "not_mine"
        "similarity": similarity,
        "total_mine": total_mine,
        "total_maybe": total_maybe,
        "total_not_mine": total_not_mine,
    }


@app.post("/api/annotate")
def annotate(req: AnnotationBody):
    annotations = load_annotations()
    autoaot = load_autoaot()

    annotations[req.path] = req.label

    # If this was in autoaot, remove it from there once resolved
    if req.path in autoaot:
        del autoaot[req.path]
        save_autoaot(autoaot)

    save_annotations(annotations)
    return {"status": "ok"}


if __name__ == "__main__":
    print("[INFO] Starting Annotator UI on http://127.0.0.1:8000")
    uvicorn.run("annotator:app", host="127.0.0.1", port=8000, reload=True)
