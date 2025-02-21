import shutil
from pathlib import Path

# Directories to clear
dirs_to_clear = ['data', 'output', 'vector_stores']

# Clear each directory
for dir_name in dirs_to_clear:
    dir_path = Path(dir_name)
    if dir_path.exists():
        shutil.rmtree(dir_path)
        dir_path.mkdir()
        print(f"Cleared {dir_name}/")

# Remove relationship visualization if it exists
vis_file = Path('document_relationships.html')
if vis_file.exists():
    vis_file.unlink()
    print("Removed document_relationships.html")