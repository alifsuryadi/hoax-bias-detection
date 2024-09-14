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

# kompas hoax news scraping
url = 'https://www.kompas.com/cekfakta/hoaks-atau-fakta'

num_page = 10 # change this to suit the purpose

# Initialize list to store data
data = []

for i in range(1, num_page):
    url_page = url+'/{}'.format(i)
    page = requests.get(url_page)
    soup = BeautifulSoup(page.text, 'html')

    # Find all news articles
    news = soup.find_all('div', 'col-bs9-3')

    for n in news:
        title_category = n.find('h1', 'cekfakta-list-title').get_text()
        link = n.find('a', 'cekfakta-list-link')['href']
        date_str = n.find('p', 'cekfakta-text-date').get_text().split(',')
        date = datetime.strptime(date_str[0].strip(), '%d/%m/%Y').date()

        # Split title and category
        part_title = title_category.split(']')
        if len(part_title) == 2:
            category = part_title[0].strip('[').strip()
            title = part_title[1].strip()

            if category == "HOAKS" and link is not None:
                # Get news contents
                get_content = requests.get(link)
                soup_content = BeautifulSoup(get_content.text, 'html')
                content = soup_content.find('div', 'read__content').find_all('p')
                paragraph = ''
                for p in content:
                    paragraph += p.get_text()

                # News information source
                is_fake = 1

                # Append data to list
                data.append([title, link, date, paragraph, is_fake])

# File name for the CSV
filename = 'dataset-kompas-hoax.csv'

# Write data to CSV file
write_to_csv(data, filename)

print(f"Data has been written to {filename}")