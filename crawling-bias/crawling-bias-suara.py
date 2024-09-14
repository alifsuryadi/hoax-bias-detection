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
        title = soup.find(class_="info").find("h1").get_text(strip=True)
        date = soup.find(class_="date-article").find("span").get_text()
        content_paragraphs = soup.find(class_="detail-content detail-berita").find_all("p")
        content = " ".join(p.get_text() for p in content_paragraphs)
        return {
            "title": title,
            "platform": "suara",
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
    for entry in soup.find_all(class_="text-list-item-y"):
        article_url = entry.find("a")["href"]
        # print(article_url)
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
        
        # Ambil href dari li terakhir
        pagination = soup.find("ul", class_="pagination")
        if pagination:
            last_page_link = pagination.find_all("li")[-1].find("a")
            next_page = last_page_link["href"] if last_page_link else None
        else:
            next_page = None

        current_page += 1
        # time.sleep(2)  # Respectful delay to avoid overwhelming the server

    return articles

def main(max_pages):
    articles = get_all_articles(base_url, max_pages)
    filename = "dataset-bias-suara.json"
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    print(f"Data has been saved to {filename}")

if __name__ == "__main__":
    max_pages = 5  # Set the number of pages you want to crawl
    base_url = "https://www.suara.com/tag/politik"
    main(max_pages)
