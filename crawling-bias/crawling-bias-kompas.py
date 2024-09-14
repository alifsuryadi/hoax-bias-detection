import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
}


def get_soup(url):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    # print(f"Successfully connected to {url}")
    return BeautifulSoup(response.content, 'html.parser')

def parse_article(article_url):
    try:
        soup = get_soup(article_url)
        title = soup.find(class_="read__title").get_text()
        date = soup.find(class_="videoKG-date").get_text()
        content_paragraphs = soup.find(class_="read__content").find_all("p")
        content = " ".join(p.get_text() for p in content_paragraphs)
        return {
            "title": title,
            "platform": "kompas",
            "link": article_url,
            "date": date,
            "content": content,
        }
    except Exception as e:
        print(f"Error parsing article {article_url}: {e}")
        return None

def parse_page(url):
    soup = get_soup(url)
    articles = []
    for entry in soup.find_all(class_="article__title article__title--medium"):
        article_url = entry.find("a")["href"]
        article = parse_article(article_url)
        if article:  # Only append if article is not None
            articles.append(article)
        # time.sleep(1)  # Respectful delay to avoid overwhelming the server
    return articles

def get_all_articles(base_url, max_pages):
    articles = []
    next_page = base_url
    current_page = 1

    while next_page and current_page <= max_pages:
        print(f"Crawling page {current_page}")
        articles.extend(parse_page(next_page))
        soup = get_soup(next_page)
        next_button = soup.find(class_="paging__link paging__link--next")
        next_page = next_button["href"] if next_button else None
        current_page += 1
        # time.sleep(2)  # Respectful delay to avoid overwhelming the server
    
    return articles

def main(max_pages):
    articles = get_all_articles(base_url, max_pages)
    filename = "dataset-bias-kompas.json"
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    print(f"Data has been saved to {filename}")

if __name__ == "__main__":
    max_pages = 5  # Set the number of pages you want to crawl
    base_url = "https://www.kompas.com/tag/politik"
    main(max_pages)
