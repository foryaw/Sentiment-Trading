
import os
from nltk.sentiment import SentimentIntensityAnalyzer
import csv

# The folder containing the earnings call in txt format
transcript_dir = 'C:/Users/user/Desktop/natural language processing/transcripts'


# Importing the FinBERT analyser
import nltk
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)


# Create a csv storing the sentiment for each company
with open('sp100.csv', mode='a', newline='', encoding='iso-8859-1') as csv_file:
    writer = csv.writer(csv_file)

    if os.stat('sentiment.csv').st_size == 0:
        writer.writerow(['ticker', 'File', 'positive', 'neutral', 'negative'])

    # Loop through each earnings call txt, generate the sentiment
    for folder in os.listdir(transcript_dir):
        for file_name in os.listdir(transcript_dir + os.sep + folder):
            print(n)
            print(folder, file_name)
            if file_name.endswith('.txt') and int(file_name[0:4]) > 2019:
                with open(transcript_dir + os.sep + folder + os.sep + file_name, 'r', encoding='iso-8859-1') as transcript_file:

                    try:
                        transcript_text = transcript_file.read()
                        sentences = nltk.sent_tokenize(transcript_text)



                        results = nlp(sentences)

                        print(results)

                        positive = 0
                        neutral = 0
                        negative = 0

                        for item in results:
                            if item['label'] == 'Positive':
                                positive += 1
                            elif item['label'] == 'Neutral':
                                neutral += 1
                            elif item['label'] == 'Negative':
                                negative += 1

                        print(f"Positive count: {positive}")
                        print(f"Neutral count: {neutral}")
                        print(f"Negative count: {negative}")

                        # Write the keyword counts to the CSV file
                        row = [folder] + [file_name] + [positive] + [neutral] + [negative]
                        writer.writerow(row)
                        csv_file.flush()

                        n = n + 1

                    except:
                        row = [folder] + [file_name] + ["error"] + ["error"] + ["error"]
                        writer.writerow(row)
                        csv_file.flush()
















