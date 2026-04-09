import os

train_file = r"D:\python projects\clone\xtts-finetune-webui\finetune_models\dataset\metadata_train.csv"
eval_file = r"D:\python projects\clone\xtts-finetune-webui\finetune_models\dataset\metadata_eval.csv"
wavs_dir = r"D:\python projects\clone\TTS_Dataset\wavs"

def fix_csv(filepath):
    if not os.path.exists(filepath):
        print(f"{filepath} not found")
        return
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if lines and lines[0].startswith("audio_file"):
        print(f"Header already exists in {filepath}")
        return
    
    new_lines = ["audio_file|text|text_normalized\n"]
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) >= 3:
            wav_name = parts[0]
            if not os.path.isabs(wav_name):
                # make an absolute path
                abs_path = os.path.join(wavs_dir, wav_name)
                # Ensure the path contains forward slashes for cross-platform compatibility just in case
                parts[0] = abs_path.replace("\\", "/")
            new_lines.append("|".join(parts) + "\n")
            
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"Fixed {filepath}")

fix_csv(train_file)
fix_csv(eval_file)
