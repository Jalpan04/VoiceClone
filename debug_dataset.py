import sys
import os

# Use the project root as working dir
project_root = r"D:\python projects\clone\xtts-finetune-webui"
sys.path.append(project_root)

import torch
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset

# Paths
train_csv = os.path.join(project_root, "finetune_models", "dataset", "metadata_train.csv")
eval_csv = os.path.join(project_root, "finetune_models", "dataset", "metadata_eval.csv")
vocab_file = os.path.join(project_root, "base_models", "v2.0.2", "vocab.json")
config_json = os.path.join(project_root, "base_models", "v2.0.2", "config.json")

language = "hi"

# Mock config
config_dataset = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="ft_dataset",
    path=os.path.dirname(train_csv),
    meta_file_train=train_csv,
    meta_file_val=eval_csv,
    language=language,
)

model_args = GPTArgs(
    max_conditioning_length=132300,
    min_conditioning_length=66150,
    debug_loading_failures=True,  # Turn this on to see errors
    max_wav_length=255995,
    max_text_length=200,
    mel_norm_file="mock",
    dvae_checkpoint="mock",
    xtts_checkpoint="mock",
    tokenizer_file=vocab_file,
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
)

class MockConfig:
    def __init__(self):
        self.model_args = model_args
        self.training_seed = 42

print("Loading samples...")
train_samples, eval_samples = load_tts_samples(
    [config_dataset],
    eval_split=True,
)

# Fix language injection (like we did in gpt_train.py)
for s in train_samples: s["language"] = language
for s in eval_samples: s["language"] = language

print(f"Loaded {len(train_samples)} samples")

print("Initializing tokenizer...")
tokenizer = VoiceBpeTokenizer(vocab_file)

print("Initializing dataset...")
dataset = XTTSDataset(MockConfig(), train_samples, tokenizer, 22050, is_eval=False)

print("Attempting to get item 0...")
try:
    item = dataset[0]
    print("Success! Item loaded.")
    print(f"Keys: {item.keys()}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
