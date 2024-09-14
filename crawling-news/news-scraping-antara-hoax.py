import csv
import time
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

# ANTARA Fake News Scraping
url = 'https://www.antaranews.com/tag/cek-fakta/'

num_page = 10 # change this to suit the purpose

# Initialize list to store data
data = []

for i in range(1, num_page):
    try:
        url_page = url + '{}'.format(i)
        page = requests.get(url_page, timeout=10)
        page.raise_for_status()  # Raise an HTTPError for bad responses
        soup = BeautifulSoup(page.text, 'html.parser')

        print(f"Processing page {i}")

        # Find all news articles
        news = soup.find_all('div', 'card__post')

        for n in news:
            title = n.find('div', 'card__post__title').find('a').get_text()
            link = n.find('div', 'card__post__title').find('a')['href']

            # Split title and category
            title_first_space_index = title.find(' ')
            category = title[:title_first_space_index]
            title_news = title[title_first_space_index + 1:].strip()

            if (category == "Hoaks!" or category == "Disinformasi!") and link is not None:
                # Get news contents
                try:
                    get_content = requests.get(link, timeout=10)
                    get_content.raise_for_status()  # Raise an HTTPError for bad responses
                    soup_content = BeautifulSoup(get_content.text, 'html.parser')
                    date = ""
                    content = soup_content.find('div', 'wrap__article-detail-content').get_text()

                    # News information source
                    is_fake = 1

                    # Append data to list
                    data.append([title_news, link, date, content, is_fake])
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching content for {link}: {e}")

        # Delay to avoid overwhelming the server
        time.sleep(2)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching page {i}: {e}")

# File name for the CSV
filename = 'dataset-antara-hoax.csv'

# Write data to CSV file
write_to_csv(data, filename)

print(f"Data has been written to {filename}")