import ebooklib
from ebooklib import epub

def epub_to_txt(epub_file, txt_file):
    """
    Convert an EPUB file to a TXT file.

    Args:
        epub_file (str): The path to the EPUB file.
        txt_file (str): The path to the output TXT file.
    """
    # Open the EPUB file
    book = epub.read_epub(epub_file)

    # Initialize the output text
    output_text = ""

    # Iterate over the documents in the EPUB file
    for doc in book.get_items():
        # Check if the document is a text document
        if doc.get_type() == ebooklib.ITEM_DOCUMENT:
            # Add the text to the output text
            output_text += doc.get_content().decode("utf-8") + "\n\n"

    # Remove HTML tags from the output text
    import re
    output_text = re.sub(r'<.*?>', '', output_text)

    # Write the output text to the TXT file
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(output_text)

epub_file = 'f:\Эйвельманс Бернар. Следы невиданных зверей - royallib.com.epub'
output_file = os.path.splitext(epub_file)[0] + ".txt"
# Example usage
epub_to_txt(epub_file, output_file)