"""
The module for receiving geodata from the Flickr website for St. Petersburg
"""
import requests
import json
import pandas as pd


def getpage(page, api_key, lat, lon, s, headers):
    """
    The function receives information from this page.
    :param page: page number
    :param api_key: api-key
    :param lat: latitude
    :param lon: longitude
    :param s: session
    :param headers: http header
    :return: total pages, list of photo data sets
    """
    url = 'https://api.flickr.com/services/rest/'   # url api
    query = {
                'method': 'flickr.photos.search',
                'api_key': api_key,
                'accuracy': 11,
                'content_type': 1,
                'has_geo': '1',
                'lat': lat,
                'lon': lon,
                'extras': 'description,date_upload,date_taken,geo,tags',
                'per_page': 100,
                'page': page,
                'format':'json',
            }   # параметры get запроса

    r = s.get(url, params=query, allow_redirects=True, headers=headers)
    json_string = r.text.replace('jsonFlickrApi(', '')
    js = json.loads(json_string[:-1])
    return js['photos']['pages'], js['photos']['photo']


def save_csv(photos):
    """
    The function saves data in .csv format.
    :param photos: dataset list
    :return: None
    """
    df = pd.DataFrame.from_dict(photos)
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    df.to_csv('geoflickr_spt.csv')


def main():
    api_key = '*******************************'    #api-key
    headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)'
                              ' snap Chromium/70.0.3538.67 Chrome/70.0.3538.67 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8;'
                          'application/json',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Pragma': 'no-cache',
                'Cache-Control': 'no-cache',
                'x-requested-with': 'XMLHttpRequest',
                'Content-Type': 'application/json;charset=utf-8',
            }
    s = requests.session()
    s.headers = headers
    lat, lon = 59.9386300, 30.3141300  # St. Peterburg
    pages, _ = getpage(1, api_key, lat, lon, s, headers)
    photos = []

    for page in range(1, pages):
        _, p = getpage(page, api_key, lat, lon, s, headers)
        photos.extend(p)
    save_csv(photos)


if __name__ == '__main__':
    main()