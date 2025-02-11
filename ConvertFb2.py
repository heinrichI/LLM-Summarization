import xml.etree.ElementTree as ET
import re

def clean_text(text):
    """Remove extra whitespaces and clean up text."""
    if text is None:
        return ""
    # Remove multiple whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespaces
    return text.strip()

def extract_text_from_fb2(fb2_path, txt_path):
    """
    Convert FB2 file to plain text
    
    Args:
        fb2_path (str): Path to the input FB2 file
        txt_path (str): Path to the output text file
    """
    try:
        # Parse the FB2 XML
        tree = ET.parse(fb2_path)
        root = tree.getroot()

        # Namespace handling
        namespace = {'fb': 'http://www.gribuser.ru/xml/fictionbook/2.0'}
        
        # Extract title
        try:
            book_title = root.find('.//fb:book-title', namespace)
            title = clean_text(book_title.text) if book_title is not None else "Untitled"
        except:
            title = "Untitled"
        
        # Extract author
        try:
            author_first = root.find('.//fb:first-name', namespace)
            author_last = root.find('.//fb:last-name', namespace)
            author = f"{clean_text(author_first.text)} {clean_text(author_last.text)}".strip()
        except:
            author = "Unknown Author"
        
        # Extract body text
        body_texts = []
        
        # Find all sections and extract text
        sections = root.findall('.//fb:section', namespace)
        for section in sections:
            # Extract paragraphs from each section
            paragraphs = section.findall('.//fb:p', namespace)
            for para in paragraphs:
                para_text = clean_text(para.text)
                if para_text:
                    body_texts.append(para_text)
        
        # Write to text file
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            # Write metadata
            txt_file.write(f"Title: {title}\n")
            txt_file.write(f"Author: {author}\n\n")
            
            # Write body text
            txt_file.write("\n".join(body_texts))
        
        print(f"Conversion complete. Text saved to {txt_path}")
    
    except Exception as e:
        print(f"Error converting FB2 to text: {e}")

# Example usage
def main():
    fb2_file = 'f:\Эва Эвергрин и проклятие великого магистра [litres] (Джули Абэ) (Z-Library).fb2'  # Replace with your FB2 file path
    txt_file = 'f:\Эва Эвергрин и проклятие великого магистра_conv1.txt'  # Replace with desired output path
    extract_text_from_fb2(fb2_file, txt_file)

if __name__ == "__main__":
    main()