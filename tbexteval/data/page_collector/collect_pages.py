#
# collect_pages.py
# Save the pages as images given the pdf and page numbers
#
#

import os
import streamlit as st
import fitz

COLLECTION_FOLDER = "tbexteval/data/cropp_tool/images"

def parse_pg_no(pg_no: str) -> list | int:
    if ',' in pg_no:
        return [int(pg) for pg in pg_no.split(',')]
    else:
        return int(pg_no)

def main():
    st.title("Page Collector")

    file = st.file_uploader("Upload a PDF file", type=["pdf"])
    page_number = st.text_input("Enter the page number")
    dpi = st.number_input("Enter the DPI", value=275)
    save = st.button("Save Page")

    if file is not None and save:
        if not os.path.exists(COLLECTION_FOLDER):
            os.makedirs(COLLECTION_FOLDER)

        # Read the file-like object into bytes
        file_bytes = file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_number = parse_pg_no(page_number)

        if isinstance(page_number, list):
            for pg in page_number:
                pg = parse_pg_no(pg)
                page = doc[pg - 1]
                image = page.get_pixmap(dpi=dpi)
                image.save(f"{COLLECTION_FOLDER}/{file.name}_{pg}.png")
        else:
            page = doc[page_number - 1]
            image = page.get_pixmap(dpi=dpi)
            image.save(f"{COLLECTION_FOLDER}/{file.name}_{page_number}.png")

        st.success("Pages saved successfully!")

if __name__ == "__main__":
    main()