import requests
from bs4 import BeautifulSoup
import pandas as pd
import json

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
}

# Replace this with your actual cookies
cookies = {
    "OTZ": "7574064_28_28__28_",
    "recaptcha-ca-e": "AaGzOmfDDZzLVTCxH7pggA3jvD7-Lwmy8bGW_qlmBfk2yAg57GsjSEWzBSk1Uo0y_FvVqCxvJV5aCiUjP7Rm6FvP8MWSilsG95NPWZabON-V03pGwfuWuVarANS7OZaypF4r0cBVICt2r0on_2_i:U=253376c6a0000000",
}

def get_soup(url):
    response = requests.get(url, headers=headers, cookies=cookies)
    response.raise_for_status()
    # print(f"Successfully connected to {url}")
    return BeautifulSoup(response.content, 'html.parser')

def parse_article(article_url):
    try:
        soup = get_soup(article_url)
        title = soup.find(class_="entry-title").get_text()
        date = soup.find(class_="entry-meta-date updated").find("a").get_text()
        content_paragraphs = soup.find(class_="entry-content mh-clearfix").find_all("p")
        content = "\n".join(p.get_text() for p in content_paragraphs)
        return {
            "title": title,
            "link": article_url,
            "date": date,
            "content": content,
            "is_fake": 1
        }
    except Exception as e:
        print(f"Error parsing article {article_url}: {e}")
        return None

def parse_page(url):
    soup = get_soup(url)
    articles = []
    for entry in soup.find_all(class_="entry-title mh-loop-title"):
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
        next_button = soup.find(class_="next page-numbers")
        next_page = next_button["href"] if next_button else None
        current_page += 1
        # time.sleep(2)  # Respectful delay to avoid overwhelming the server
    
    return articles

def main(max_pages):
    articles = get_all_articles(base_url, max_pages)
    filename = "dataset-turnbackhoax.json"
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    print(f"Data has been saved to {filename}")

if __name__ == "__main__":
    max_pages = 5  # Set the number of pages you want to crawl
    base_url = "https://turnbackhoax.id/page/1/"
    main(max_pages)
