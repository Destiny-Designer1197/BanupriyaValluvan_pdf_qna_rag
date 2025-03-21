import os
import glob
from zipfile import ZipFile
import PyPDF2
from config import *


def extract_text_from_pdfs(folder):
    """
    This function extracts text from all PDF files within a specified zip folder.

    Parameters:
    folder (str): The path to the zip folder containing the PDF files.

    Returns:
    str: A string containing the extracted text from all the PDF files.
    """
    text = ""
  
    for file in glob.glob(os.path.join(folder, "*.pdf")):
        reader = PyPDF2.PdfReader(file)  # Create a PDF reader object
        num_pages = len(reader.pages)  # Get the number of pages

    # Loop through each page and extract text
        for page_num in range(num_pages):
            page = reader.pages[page_num]  # Get a specific page
            text += page.extract_text() 

    return text
