import requests
from bs4 import BeautifulSoup
import time
import re
import json

def clean_joke(html_joke):
    # Replace <br /> with newlines and remove other HTML tags
    text = re.sub(r'<br\s*/?>', '\n', html_joke)
    text = BeautifulSoup(text, 'html.parser').get_text()
    return text.strip()

def scrape_jokes(source_url: str, num_pages: list = range(1, 10), delay=1.5):
    base_url = source_url
    all_jokes = []

    end_page = -1
    for page in num_pages:
        url = base_url + str(page)
        print(f"Scraping page {page}: {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        joke_blocks = soup.find_all('div', class_='container joke-here')

        for block in joke_blocks:
            joke_html = str(block).split('<div class="about">')[0]
            cleaned = clean_joke(joke_html)
            all_jokes.append({"joke": cleaned})

        time.sleep(delay)
        end_page = page
     
    scraped_range = None   
    if end_page != num_pages[-1]:
        print("> COULDN'T PARSE ALL PAGES")
        scraped_range = (num_pages[0], end_page)
    else:
        print("> PARSED ALL PAGES")
        scraped_range = (num_pages[0], num_pages[-1])

    return all_jokes, scraped_range

def save_jokes_to_txt(jokes, filename="polish_jokes.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for joke in jokes:
            f.write(joke + "\n\n---\n\n")
def save_jokes_to_json(jokes, scrapped_range: tuple):
    path_lambda = lambda scrapped_range: f"polish_jokes_{scrapped_range[0]}-{scrapped_range[1]}.json"
    filename = path_lambda(scrapped_range)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(jokes, f, ensure_ascii=False, indent=2)
        
# Example usage
if __name__ == "__main__":
    source_url = "https://perelki.net/?ps="
    
    jokes, scrapped_range = scrape_jokes(
        source_url=source_url,
        num_pages=range(301, 401),
        delay=2.5,
    )
    save_jokes_to_json(jokes, scrapped_range)
    print(f"Scraped {len(jokes)} jokes.")
