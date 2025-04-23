import requests
from bs4 import BeautifulSoup
import time
import json

def clean_joke(joke_html):
    # Convert <br> to \n and remove tags
    return BeautifulSoup(joke_html.replace("<br />", "\n"), "html.parser").get_text().strip()

def scrape_jeja_jokes(base_url: str, page_range: range, delay=1.5):
    all_jokes = []
    last_page = -1

    for page in page_range:
        url = f"{base_url}{page}.html"
        print(f"Scraping page {page}: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"> Failed to load page {page}")
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        joke_divs = soup.find_all('div', class_='dow-left-text')

        if not joke_divs:
            print(f"> No jokes found on page {page}. Ending early.")
            break

        for div in joke_divs:
            joke_html = str(div)
            joke = clean_joke(joke_html)
            all_jokes.append({"joke": joke})

        last_page = page
        time.sleep(delay)

    scrape_range = (page_range[0], last_page)
    return all_jokes, scrape_range

def save_jokes_to_json(jokes, scrape_range, filename_prefix="jeja_jokes"):
    filename = f"{filename_prefix}_{scrape_range[0]}-{scrape_range[1]}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(jokes, f, ensure_ascii=False, indent=2)

# Example usage
if __name__ == "__main__":
    base_url = "https://dowcipy.jeja.pl/nowe,0,0,"
    jokes, scraped_range = scrape_jeja_jokes(base_url, page_range=range(1, 914), delay=2)
    save_jokes_to_json(jokes, scraped_range)
    print(f"Scraped {len(jokes)} jokes.")
