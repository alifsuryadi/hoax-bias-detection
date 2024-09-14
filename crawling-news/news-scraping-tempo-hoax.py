import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Function to write data to CSV file

def write_to_csv(data, filename):
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header row
        writer.writerow(['title', 'link', 'date', 'content', 'is_fake'])
        # Write data rows
        for row in data:
            writer.writerow(row)

# Tempo Fake News Scraping
year = 2021
url = 'https://cekfakta.tempo.co/2021/' # change year for crawling

# Initialize list to store data
data = []

for i in range(1, 13):
  if (i < 10):
    url_page = url+'0{}'.format(i)
  else:
    url_page = url+'{}'.format(i)
  page = requests.get(url_page)
  soup = BeautifulSoup(page.text, 'html')

  print(i)

  # Find all news articles
  news = soup.find_all('article', 'text-card')

  for n in news:
    title = n.find('a').get_text()
    link = n.find('a')['href']

    # Split title and category
    title_first_space_index = title.find(',')
    category = title[:title_first_space_index]
    title_news = title[title_first_space_index + 1:].strip()

    if (category == "Menyesatkan" or category == "Keliru") and link is not None:
      # Get news contents
      if link.startswith("//"):
        news_link = "http:" + link
      else:
        news_link = link
      get_content = requests.get(news_link)
      soup_content = BeautifulSoup(get_content.text, 'html')
      date = soup_content.find('div', 'detail-title').find('p', 'date margin-bottom-sm').get_text()
      content = soup_content.find('div', 'detail-in').find_all('p')
      paragraph = ''
      for p in content:
        #p.span.unwrap()
        paragraph += p.get_text()

      # News information source
      is_fake = 1

      # Append data to list
      data.append([title_news, news_link, date, paragraph, is_fake])

# File name for the CSV
filename_draft = f'dataset-tempo-hoax-draft-{year}.csv'
filename = f'dataset-tempo-hoax-{year}.csv'

# Write data to CSV file
write_to_csv(data, filename_draft)

# Read CSV file
df = pd.read_csv(f'dataset-tempo-hoax-draft-{year}.csv')

# remove duplicate data
df_unique = df.drop_duplicates(subset='title', keep='first')

# save new data
df_unique.to_csv(filename, index=False)

print(f"Data has been written to {filename}")