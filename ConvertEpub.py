import zipfile
from xml.etree import ElementTree as ET
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import os
import re

def extract_text_from_epub(epub_path):
    # Open the EPUB file
    book = epub.read_epub(epub_path)
    
    # Initialize an empty string to store text
    book_text = []
    
    # Iterate through each item in the book
    for item in book.get_items():
        # Check if the item is an HTML document (XHTML)
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Parse the HTML content
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            
            # Extract text from the parsed HTML
            text = soup.get_text()

            
            # Replace multiple newlines with single newline
            cleaned_text = re.sub(r'\n+', '\n', text)
            
            # Add the text to our list
            book_text.append(cleaned_text)
    
    # Join all extracted text
    joined = '\n'.join(book_text)
    return joined

def normalize_path(path):
    """Normalize path to use forward slashes."""
    return path.replace('\\', '/')

def ConvertFromEpub(epub_path: str):
    with zipfile.ZipFile(epub_path, 'r') as zipf:
        # Print all files in the ZIP for debugging
        print("Files in ZIP:", zipf.namelist())

        # Step 1: Locate and parse container.xml
        container_path = 'META-INF/container.xml'
        if container_path not in zipf.namelist():
            raise FileNotFoundError(f"'{container_path}' not found in the EPUB archive.")

        with zipf.open(container_path) as container_file:
            container_xml = container_file.read()
            container_tree = ET.fromstring(container_xml)
            rootfile_element = container_tree.find(
                './/{urn:oasis:names:tc:opendocument:xmlns:container}rootfile'
            )
            if rootfile_element is None:
                raise ValueError("No <rootfile> element found in container.xml.")

            opf_path = rootfile_element.attrib.get('full-path')
            if not opf_path:
                raise ValueError("'full-path' attribute not found in <rootfile> element.")

        opf_path = normalize_path(opf_path)
        print(f"OPF path: {opf_path}")

        # Step 2: Parse content.opf
        with zipf.open(opf_path) as opf_file:
            opf_xml = opf_file.read()
            opf_tree = ET.fromstring(opf_xml)

            namespaces = {
                'opf': 'http://www.idpf.org/2007/opf',
                'dc': 'http://purl.org/dc/elements/1.1/'
            }

            manifest = {}
            for item in opf_tree.findall('opf:manifest/opf:item', namespaces):
                manifest[item.attrib['id']] = item.attrib

            spine = opf_tree.find('opf:spine', namespaces)
            if spine is None:
                raise ValueError("No <spine> element found in content.opf.")

            spine_itemrefs = spine.findall('opf:itemref', namespaces)
            if not spine_itemrefs:
                raise ValueError("No <itemref> elements found in <spine>.")

        print(f"Number of spine items: {len(spine_itemrefs)}")

        # Step 3: Extract and concatenate text
        text_content = []

        for itemref in spine_itemrefs:
            idref = itemref.attrib.get('idref')
            if not idref:
                continue

            item = manifest.get(idref)
            if not item:
                continue

            href = normalize_path(item.get('href', ''))
            media_type = item.get('media-type', '')

            if media_type not in ['application/xhtml+xml', 'text/html', 'application/html+xml']:
                continue

            # Construct content path
            opf_dir = os.path.dirname(opf_path)
            content_path = os.path.normpath(os.path.join(opf_dir, href)).replace('\\', '/')

            # Try different path variations
            possible_paths = [
                content_path,
                content_path.split('/', 1)[-1],  # Remove first directory
                href
            ]

            found_path = None
            for path in possible_paths:
                if path in zipf.namelist():
                    found_path = path
                    break

            if not found_path:
                print(f"Warning: Content file not found for {href}")
                continue

            print(f"Processing: {found_path}")

            with zipf.open(found_path) as content_file:
                content_html = content_file.read()
                soup = BeautifulSoup(content_html, 'html.parser')
                for element in soup(['script', 'style', 'nav']):
                    element.decompose()
                text = soup.get_text(separator='\n')
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                clean_text = '\n'.join(lines)

                # **Cleaning Steps Start Here**
                # Replace non-breaking spaces with regular spaces
                # clean_text = clean_text.replace('\xa0', ' ')
                # Replace en dashes with hyphens (optional)
                # clean_text = clean_text.replace('–', '-')
                # Optionally, you can use more replacements or regular expressions
                # import re
                # clean_text = re.sub(r'–\xa0', '-', clean_text)
                # **Cleaning Steps End Here**
                import unicodedata
                clean_text = unicodedata.normalize('NFKC', clean_text)

                text_content.append(clean_text)

        print(f"Total text content length: {len(''.join(text_content))}")

    return "\n".join(text_content)
    # return text_content

# def epub_to_txt(epub_path, output_path):
#     try:
#         text_content = ConvertFromEpub(epub_path)
#         # Step 4: Write to output file
#         with open(output_path, 'w', encoding='utf-8') as txt_file:
#             txt_file.write('\n\n'.join(text_content))

#         print(f"Successfully converted '{epub_path}' to '{output_path}'.")

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         raise

# # Example Usage
# if __name__ == "__main__":
#     import argparse

#     # parser = argparse.ArgumentParser(description="Convert EPUB to TXT.")
#     # parser.add_argument("epub_path", help="Path to the input EPUB file.")
#     # parser.add_argument("output_path", nargs='?', help="Path for the output TXT file.")

#     # args = parser.parse_args()

#     # epub_file = args.epub_path
#     epub_file = 'f:\Эйвельманс Бернар. Следы невиданных зверей - royallib.com.epub'
#     # txt_file = args.output_path if args.output_path else os.path.splitext(epub_file)[0] + ".txt"
#     txt_file = os.path.splitext(epub_file)[0] + ".txt"

#     epub_to_txt(epub_file, txt_file)
