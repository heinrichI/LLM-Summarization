import xml.etree.ElementTree as ET
import os
import argparse
import unicodedata

def extract_text_from_fb2(fb2_file):
    """
    Extract plain text from an FB2 file
    """
    try:
        # Parse the FB2 XML file
        tree = ET.parse(fb2_file)
        root = tree.getroot()

        # Define XML namespaces
        namespaces = {
            'fb': 'http://www.gribuser.ru/xml/fictionbook/2.0'
        }

        # Initialize text content
        full_text = []

        # Extract title
        try:
            book_title = root.find('.//fb:book-title', namespaces)
            if book_title is not None:
                full_text.append(book_title.text.strip())
                full_text.append('\n')
        except Exception:
            pass

        # Extract author information
        try:
            authors = root.findall('.//fb:author', namespaces)
            for author in authors:
                first_name = author.find('fb:first-name', namespaces)
                last_name = author.find('fb:last-name', namespaces)
                if first_name is not None and last_name is not None:
                    full_text.append(f"{first_name.text} {last_name.text}")
            full_text.append('\n\n')
        except Exception:
            pass

        # Extract body text
        body_sections = root.findall('.//fb:section', namespaces)
        for section in body_sections:
            # Extract paragraphs
            paragraphs = section.findall('.//fb:p', namespaces)
            for paragraph in paragraphs:
                # Remove nested elements and get text
                text = extract_paragraph_text(paragraph)
                if text:
                    full_text.append(text)
                    full_text.append('\n')

        return '\n'.join(full_text)

    except Exception as e:
        print(f"Error processing {fb2_file}: {e}")
        return None

def extract_paragraph_text(paragraph):
    """
    Extract text from a paragraph, removing nested elements
    """
    text_parts = []
    
    # Handle text directly in paragraph
    if paragraph.text:
        text_parts.append(paragraph.text.strip())
    
    # Handle nested elements
    for child in paragraph:
        # Get text of child element
        if child.text:
            text_parts.append(child.text.strip())
        
        # Get tail text (text after element)
        if child.tail:
            text_parts.append(child.tail.strip())
    
    return unicodedata.normalize('NFKC', ' '.join(text_parts))

def convert_fb2_to_txt(input_path, output_path=None):
    """
    Convert FB2 file or directory of FB2 files to text
    """
    # Determine input and output paths
    if output_path is None:
        output_path = os.path.dirname(input_path)

    # Handle single file
    if os.path.isfile(input_path):
        if input_path.lower().endswith('.fb2'):
            txt_content = extract_text_from_fb2(input_path)
            if txt_content:
                # Generate output filename
                txt_filename = os.path.splitext(os.path.basename(input_path))[0] + '.txt'
                txt_filepath = os.path.join(output_path, txt_filename)
                
                # Write text to file
                with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(txt_content)
                print(f"Converted: {input_path} -> {txt_filepath}")

    # Handle directory
    elif os.path.isdir(input_path):
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Process all FB2 files in directory
        for filename in os.listdir(input_path):
            if filename.lower().endswith('.fb2'):
                fb2_filepath = os.path.join(input_path, filename)
                convert_fb2_to_txt(fb2_filepath, output_path)

def main():
    # Set up argument parsing
    # parser = argparse.ArgumentParser(description='Convert FB2 files to plain text')
    # parser.add_argument('input', help='Input FB2 file or directory')
    # parser.add_argument('-o', '--output', help='Output directory (optional)', default=None)
    
    # # Parse arguments
    # args = parser.parse_args()

    # Convert FB2 to text
    # convert_fb2_to_txt(args.input, args.output)
    convert_fb2_to_txt('f:\Эва Эвергрин и проклятие великого магистра [litres] (Джули Абэ) (Z-Library).fb2')

if __name__ == '__main__':
    main()