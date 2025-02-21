import shutil
from pathlib import Path

# First, ensure preprocess directory exists
preprocess_dir = Path('preprocess')
preprocess_dir.mkdir(exist_ok=True)

# Move PDFs from finished back to preprocess
finished_dir = Path('finished')
if finished_dir.exists():
    print("\nMoving PDFs from finished back to preprocess folder...")
    for pdf_file in finished_dir.glob('*.pdf'):
        dest_path = preprocess_dir / pdf_file.name
        try:
            pdf_file.rename(dest_path)
            print(f"Moved {pdf_file.name} back to preprocess/")
        except Exception as e:
            print(f"Error moving {pdf_file.name}: {str(e)}")
    
    # Remove finished directory after moving files
    try:
        shutil.rmtree(finished_dir)
        print("Removed finished directory")
    except Exception as e:
        print(f"Error removing finished directory: {str(e)}")

# Directories to clear
print("\nClearing previous processing data...")
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

print("\nCleanup complete! PDFs are ready for reprocessing in the preprocess folder.")