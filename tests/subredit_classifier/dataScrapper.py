import requests
import sys
from bs4 import BeautifulSoup

subreddits = [
    'worldnews',
    'AskReddit',
    'pics',
    'politics',
    'gaming',
    'programming',
    'dankmemes'
]

# Todo: Create a reddit api account

base_url = "http://www.reddit.com/r/"
headers = {
    'User-Agent': 'Mozilla/5.0 (Android 4.4; Mobile; rv:41.0) Gecko/41.0 Firefox/41.0'
}

number_posts_per_subreddit = 1000

def main(args):
    for subredit in subreddits:
        full_url = base_url + subredit
        page = requests.get(full_url, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
        print(soup.prettify())


if __name__ == "__main__":
    main(sys.argv[1:])
