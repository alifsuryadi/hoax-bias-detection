import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# function to write data to CSV file
def write_to_csv(data, filename):
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header row
        writer.writerow(['title', 'link', 'date', 'content', 'is_fake'])
        # Write data rows
        for row in data:
            writer.writerow(row)

# kompas news scraping
url = 'https://indeks.kompas.com/?site=news'

num_page = 5 # change this to suit the purpose

# Initialize list to store data
data = []

for i in range(1, num_page):
    url_page = url+'&page={}'.format(i)
    page = requests.get(url_page)
    soup = BeautifulSoup(page.text, 'html')

    # Find all news articles
    news = soup.find_all('div', 'articleItem')

    for n in news:
        title = n.find('h2').get_text()
        link = n.find('a', 'article-link')['href']
        date_str = n.find('div', 'articlePost-date').get_text()
        date = datetime.strptime(date_str, '%d/%m/%Y').date()

        if(link != ''):
            # Get news contents
            get_content = requests.get(link)
            soup_content = BeautifulSoup(get_content.text, 'html')
            content = soup_content.find('div', 'read__content').find_all('p')
            paragraph = ''
            for p in content:
                paragraph += p.get_text()

            # News information source
            is_fake = 0

            # Append data to list
            data.append([title, link, date, paragraph, is_fake])

# File name for the CSV
filename = 'dataset-kompas.csv'

# Write data to CSV file
write_to_csv(data, filename)

print(f"Data has been written to {filename}")