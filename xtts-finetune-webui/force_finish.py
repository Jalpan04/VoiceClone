import os
import requests
import time

def force_download(url, dest):
    print(f"Force downloading {os.path.basename(dest)}...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    for i in range(100): # 100 retries
        try:
            # Using a browser-like User-Agent to try and sneak past the ISP block
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, stream=True, timeout=20)
            if response.status_code == 200:
                with open(dest, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  SUCCESS: {os.path.basename(dest)}")
                return True
        except Exception as e:
            print(f"  Attempt {i+1} failed ({e}). Retrying...")
            time.sleep(2)
    return False

if __name__ == "__main__":
    base_dir = "base_models/v2.0.2"
    files = [
        ("https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth", f"{base_dir}/dvae.pth"),
        ("https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth", f"{base_dir}/mel_stats.pth"),
        # Also resume the main model file just in case it's missing the last few bytes
        ("https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/model.pth", f"{base_dir}/model.pth"),
    ]
    
    for url, path in files:
        force_download(url, path)
