import sys
try:
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(r"D:\python projects\clone\xtts-finetune-webui\base_models\v2.0.2\vocab.json")
    print(f"SUCCESS: Tokenizer loaded. Vocab size: {t.get_vocab_size()}")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
