import os
import requests
import time

def download_file(url, dest_path):
    print(f"Checking {os.path.basename(dest_path)}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    file_size = 0
    if os.path.exists(dest_path):
        file_size = os.path.getsize(dest_path)
        print(f"  Existing size: {file_size / (1024*1024):.2f} MB")

    retries = 0
    while True:
        try:
            headers = {}
            if file_size > 0:
                headers['Range'] = f'bytes={file_size}-'
            
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # If server doesn't support range or file is complete
            if response.status_code == 416: # Range not satisfiable (might be done)
                print("  File already complete or range error.")
                return
            
            mode = 'ab' if file_size > 0 else 'wb'
            with open(dest_path, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        file_size += len(chunk)
                        # Print progress every 10MB
                        if file_size % (10 * 1024 * 1024) < 8192:
                            print(f"  Progress: {file_size / (1024*1024):.2f} MB", end='\r')
            
            print(f"\n  Finished: {os.path.basename(dest_path)}")
            break
            
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            retries += 1
            print(f"\n  Connection lost (Retry {retries}). Resuming in 5s...")
            time.sleep(5)
            # Refresh file size before retry
            file_size = os.path.getsize(dest_path)

if __name__ == "__main__":
    base_dir = "base_models/v2.0.2"
    base_url = "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2"
    main_url = "https://huggingface.co/coqui/XTTS-v2/resolve/main"
    
    files = [
        (f"{base_url}/model.pth", f"{base_dir}/model.pth"),
        (f"{base_url}/config.json", f"{base_dir}/config.json"),
        (f"{base_url}/vocab.json", f"{base_dir}/vocab.json"),
        (f"{main_url}/dvae.pth", f"{base_dir}/dvae.pth"),
        (f"{main_url}/mel_stats.pth", f"{base_dir}/mel_stats.pth"),
        (f"{main_url}/speakers_xtts.pth", f"{base_dir}/speakers_xtts.pth"),
    ]
    
    for url, path in files:
        download_file(url, path)
    
    print("\n--- ALL DOWNLOADS COMPLETE ---")
