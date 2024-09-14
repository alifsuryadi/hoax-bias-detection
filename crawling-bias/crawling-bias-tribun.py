from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time
from bs4 import BeautifulSoup
import requests
import json

def get_soup(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return BeautifulSoup(response.content, 'html.parser')

def parse_article(article_url):
    try:
        soup = get_soup(article_url)
        title = soup.find(class_="f50 black2 f400 crimson").get_text()
        date = soup.find(class_="grey bdr3 pb10 pt10").find("span").get_text()
        content_paragraphs = soup.find(class_="side-article txt-article multi-fontsize").find_all("p")
        content = " ".join(p.get_text() for p in content_paragraphs)
        return {
            "title": title,
            "platform": "tribun",
            "url": article_url,
            "date": date,
            "content": content,
        }
    except Exception as e:
        print(f"Error parsing article {article_url}: {e}")
        return None

def perform_scrolling(driver, num_scrolls):
    for _ in range(num_scrolls):
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time.sleep(1)  # Adjust sleep time if necessary

def get_article_links(driver, url, num_scrolls):
    driver.get(url)
    perform_scrolling(driver, num_scrolls)
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    return soup.select('div.mr140 h3 a')

def scrape_articles(driver, url, num_scrolls):
    articles = get_article_links(driver, url, num_scrolls)
    article_data = []
    
    for article in articles:
        article_url = article['href']
        article_info = parse_article(article_url)
        if article_info:
            article_data.append(article_info)
        # Add a delay to avoid being blocked by the server
        # time.sleep(1)
    
    return article_data

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')  # Add this to ignore SSL errors
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    url = "https://www.tribunnews.com/nasional/politik"
    num_scrolls = 5
    article_data = scrape_articles(driver, url, num_scrolls)
    filename = "dataset-bias-tribun.json"
    save_to_json(article_data, filename)
    
    driver.quit()

if __name__ == "__main__":
    main()
