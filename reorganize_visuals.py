import os
from pathlib import Path
import shutil
import re

def reorganize_visual_elements():
    """Reorganize visual elements into document-specific folders"""
    visuals_dir = Path("output/visual_elements")
    if not visuals_dir.exists():
        print("Visual elements directory not found")
        return

    # Get all PNG files
    visual_files = list(visuals_dir.glob("*.png"))
    if not visual_files:
        print("No visual elements found")
        return

    # Group files by document name
    doc_visuals = {}
    for file_path in visual_files:
        # Extract document name from file name (everything before _visual_)
        match = re.match(r"(.+?)_visual_\d+\.png", file_path.name)
        if match:
            doc_name = match.group(1)
            if doc_name not in doc_visuals:
                doc_visuals[doc_name] = []
            doc_visuals[doc_name].append(file_path)

    # Create document folders and move files
    for doc_name, files in doc_visuals.items():
        # Create document folder
        doc_folder = visuals_dir / doc_name
        doc_folder.mkdir(exist_ok=True)
        
        # Move visual files
        for file_path in files:
            # Get the visual number
            visual_num = re.search(r"_visual_(\d+)\.png", file_path.name).group(1)
            new_name = f"visual_{visual_num}.png"
            new_path = doc_folder / new_name
            
            # Move the file
            shutil.move(str(file_path), str(new_path))
            print(f"Moved {file_path.name} to {doc_name}/{new_name}")

        # Move corresponding JSON metadata if it exists
        json_file = visuals_dir / f"{doc_name}_visuals.json"
        if json_file.exists():
            new_json_path = doc_folder / "metadata.json"
            shutil.move(str(json_file), str(new_json_path))
            print(f"Moved {json_file.name} to {doc_name}/metadata.json")

    print("\nReorganization complete!")

if __name__ == "__main__":
    reorganize_visual_elements() 