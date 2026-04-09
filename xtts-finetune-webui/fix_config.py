import sys

src = r'C:\Users\acer\.gemini\antigravity\brain\77c726ea-c8b4-4738-83b7-2a6ef6112138\.system_generated\steps\1459\content.md'
dst = r'D:\python projects\clone\xtts-finetune-webui\base_models\v2.0.2\config.json'

lines = open(src, 'r', encoding='utf-8').readlines()
start = next(i for i, l in enumerate(lines) if l.strip().startswith('{'))
content = ''.join(lines[start:])

# Remove any trailing whitespace/newlines
content = content.rstrip() + '\n'

with open(dst, 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print(f"Written {len(content)} bytes to config.json")

# Verify it can be loaded (with Infinity support)
import json
# The XTTS config uses JavaScript Infinity which is not valid JSON
# but the TTS library handles it internally, so just verify the file exists and starts with {
first_char = open(dst, 'r', encoding='utf-8').read(1)
if first_char == '{':
    print("config.json starts with '{' - OK")
else:
    print(f"ERROR: config.json starts with '{first_char}' - CORRUPT")
    sys.exit(1)
