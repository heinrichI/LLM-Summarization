import zipfile
from bs4 import BeautifulSoup
import os
import shutil

def ConvertFromEpub(epub_path: str):
    # Initialize the output text
    output_text = ""

    # Open the EPUB file as a ZIP archive
    with zipfile.ZipFile(epub_path, 'r') as zip_ref:
        # Iterate over the files in the EPUB archive
        for file in zip_ref.namelist():
            # Check if the file is an HTML or XHTML file
            if file.endswith(".html") or file.endswith(".xhtml"):
                # Read the file's content
                with zip_ref.open(file, 'r') as f:
                    # Parse the HTML file using BeautifulSoup
                    soup = BeautifulSoup(f.read().decode('utf-8'), 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    # Get the text from the HTML file
                    text = soup.get_text()
                    # Add the text to the output text
                    output_text += text + "\n\n"
    return output_text

# def epub_to_txt(epub_file, txt_file):
#     """
#     Convert an EPUB file to a TXT file.

#     Args:
#         epub_file (str): The path to the EPUB file.
#         txt_file (str): The path to the output TXT file.
#     """

#     output_text = ConvertFromEpub(epub_file)

#     # Write the output text to the TXT file
#     with open(txt_file, "w", encoding="utf-8") as f:
#         f.write(output_text)

# epub_file = 'f:\Эйвельманс Бернар. Следы невиданных зверей - royallib.com.epub'
# output_file = os.path.splitext(epub_file)[0] + ".txt"
# # Example usage
# epub_to_txt(epub_file, output_file)