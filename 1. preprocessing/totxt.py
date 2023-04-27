import os
from PyPDF2 import PdfReader
# Converting the earnings call transcript collected from Bloomberg from pdf to txt

# The folder containing the folders of earnings call transcript in pdf format
company_list = os.listdir("C:/Users/user/Desktop/natural language processing/extract earnings")

# Loop through each company folder to extract pdf
for company in company_list:
    for transcript in os.listdir("C:/Users/user/Desktop/natural language processing/extract earnings" + os.sep + company):

        # Convert the pdf to txt and stored in the folder transcripts/company
        if transcript.endswith(".pdf"):

            output_dir = os.path.join("C:/Users/user/Desktop/natural language processing/transcripts" + os.sep + company)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, transcript[:-4] + '.txt')
            with open(output_file, 'w', encoding='UTF-8') as text_file:
                reader = PdfReader(r'C:\Users\user\Desktop\natural language processing\extract earnings' + os.sep + company + os.sep + t)
                number_of_pages = len(reader.pages)

                for page_number in range(number_of_pages):
                    page = reader.pages[page_number]
                    text_file.write(page.extract_text())

