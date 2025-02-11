from typing import List
import os

def save_summarization_results(file_path: str, final_summary: str, intermediate_summaries: List[str]):
    """
    Saves the summarization results to files near the source text file.
    
    Args:
        file_path: Path to the original text file
        final_summary: The final summary text
        intermediate_summaries: List of intermediate summaries
    """
    try:
        # Get the directory and filename without extension
        file_dir = os.path.dirname(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Save final summary
        final_summary_path = os.path.join(file_dir, f"{file_name}_final_summary.txt")
        with open(final_summary_path, 'w', encoding='utf-8') as f:
            f.write(final_summary)

        # Save intermediate summaries
        intermediate_summary_path = os.path.join(file_dir, f"{file_name}_intermediate_summaries.txt")
        with open(intermediate_summary_path, 'w', encoding='utf-8') as f:
            for i, summary in enumerate(intermediate_summaries, 1):
                f.write(f"=== Intermediate Summary {i} ===\n")
                f.write(summary)
                f.write("\n\n")
        
        print(f"Final summary: {final_summary_path}")
        print(f"Intermediate summaries: {intermediate_summary_path}")
        
    except Exception as e:
        print(f"Error saving summarization results: {e}")

def ReadFile(file_path: str) -> str:
    import zipfile
    import xml.etree.ElementTree as ET
    from ConvertEpub import extract_text_from_epub
    from ConvertFb2_2 import extract_text_from_fb2
     # Determine the file extension
    file_extension = os.path.splitext(file_path)[1].lower()

      # Handle .txt files with ANSI or UTF-8 encoding
    if file_extension == '.txt':
        try:
            # First try UTF-8
            with open(file_path, "r", encoding="utf-8") as f:
                document = f.read()
        except UnicodeDecodeError:
            # If UTF-8 fails, try ANSI (Windows-1251)
            with open(file_path, "r", encoding="windows-1251") as f:
                document = f.read()
        return document
    
    # Handle .epub files
    elif file_extension == '.epub':
        text = extract_text_from_epub(file_path)

        # file_dir = os.path.dirname(file_path)
        # file_name = os.path.splitext(os.path.basename(file_path))[0]
        # # Save final summary
        # converted_path = os.path.join(file_dir, f"{file_name}_conv.txt")
        # with open(converted_path, 'w', encoding='utf-8') as f:
        #     f.write(text)

        return text

    # Handle .fb2 files
    elif file_extension == '.fb2':
        return extract_text_from_fb2(file_path)

    # Handle .zip files
    elif file_extension == '.zip':
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # List all files in the zip archive
            zip_files = zip_ref.namelist()
            
            # Check if any file in the archive has a .fb2 extension
            fb2_files = [f for f in zip_files if f.lower().endswith('.fb2')]
            
            if not fb2_files:
                raise ValueError("No .fb2 file found in the zip archive.")
            
            # Read the first .fb2 file found in the archive
            fb2_file = fb2_files[0]
            with zip_ref.open(fb2_file) as fb2:
                # Extract text from the FB2 content
                return extract_text_from_fb2(fb2)
    
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
