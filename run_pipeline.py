import sys
import subprocess
import json
import os
from pathlib import Path

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check_annotations():
    if not os.path.exists("annotations.json"):
        return 0
    with open("annotations.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    mine_count = sum(1 for v in data.values() if v == "mine")
    return mine_count

def run_script(path):
    print(f"\n>> Running {path}...")
    try:
        # Use the same python executable to ensure environment is preserved
        subprocess.run([sys.executable, path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Script {path} failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print(f"\n[ABORTED] Script {path} was interrupted by user.")
    print(">> Done.\n")

def main():
    while True:
        mine_count = check_annotations()
        
        print_header("SPEAKER DIARIZATION & ANNOTATION PIPELINE")
        print("Welcome to the automated speaker sorting pipeline.")
        print("This tool will guide you through extracting your target voice.")
        print(f"\n[Current Status]: {mine_count} confirmed 'mine' folders found.")
        print("\nWhat would you like to do?")
        print("  1. Launch User Interface (Manual Annotation)")
        print("     -> Do this first to seed a few 'mine' clips (aim for 15+)")
        print("     -> Use this later to verify AI matches (Maybes/Strays)")
        print("\n  2. Phase 1: Auto Annotate (Find obvious matches)")
        print("     -> Requires UI to be closed. Finds matches using your seed.")
        print("\n  3. Prune Empty Folders")
        print("     -> Cleans up folders that have no usable audio.")
        print("\n  4. Phase 2: Gold Standard Stray Search")
        print("     -> Rescans everything rejected to find hidden matches")
        print("     -> (Best used after reviewing all maybes)")
        print("\n  5. Final Export")
        print("     -> Copies all your 'mine' clips into a single folder")
        print("\n  0. Exit")
        
        choice = input("\nEnter choice (0-5): ").strip()
        
        if choice == "0":
            print("Exiting pipeline. Goodbye!")
            break
            
        elif choice == "1":
            print_header("LAUNCHING UI")
            print("The Annotation Web UI is starting...")
            print("Open http://127.0.0.1:8000 in your browser.")
            print("Press CTRL+C in this terminal when you are done to return to the menu.")
            try:
                subprocess.run([sys.executable, "annotator.py"])
            except KeyboardInterrupt:
                pass
                
        elif choice == "2":
            print_header("PHASE 1: AUTO ANNOTATOR")
            if mine_count < 5:
                print("⚠️ WARNING: You have very few 'mine' clips. The AI might not be accurate.")
                confirm = input("Continue anyway? (y/n): ")
                if confirm.lower() != 'y':
                    continue
            run_script(os.path.join("src", "auto_annotator.py"))
            
        elif choice == "3":
            print_header("PRUNING EMPTY FOLDERS")
            run_script(os.path.join("tools", "prune_empty_folders.py"))
            
        elif choice == "4":
            print_header("PHASE 2: GOLD STANDARD STRAY SEARCH")
            if mine_count < 20:
                print("⚠️ WARNING: You should have a large 'mine' dataset before sweeping for strays.")
                confirm = input("Continue anyway? (y/n): ")
                if confirm.lower() != 'y':
                    continue
            run_script(os.path.join("tools", "gold_standard_check.py"))
            
        elif choice == "5":
            print_header("FINAL EXPORT")
            run_script(os.path.join("tools", "export_mine_clips.py"))
            
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
