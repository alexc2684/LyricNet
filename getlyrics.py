import socket
import json
import requests
import re
from bs4 import BeautifulSoup

CLIENT_TOKEN = "FAlLLpDDvipZuMDOBM07qRkDgw_kQs3_l3KYJuMsXCE3EJIE77MdWC8FP2x2ieCO"
URL = "http://api.genius.com"

def getRequest(url, headers):
    return requests.request(url=url, headers=headers, method="GET")

    # page = 1
    # while page < 5:
    #     url = "http://api.genius.com/search?q=" + urllib2.quote(query) + "&page=" + str(page)
    #     req = urllib2.Request(url)
    #     req.add_header("Authorization", "Bearer " + CLIENT_TOKEN)
    #     req.add_header("User-Agent", "curl/7.9.8 (i686-pc-linux-gnu) libcurl 7.9.8 (OpenSSL 0.9.6b) (ipv6 enabled)")



headers = {"Authorization": "Bearer " + CLIENT_TOKEN, "User-Agent": "curl/7.9.8 (i686-pc-linux-gnu) libcurl 7.9.8 (OpenSSL 0.9.6b) (ipv6 enabled)"}
# search_url = URL + "/songs/" + "70324"
# req = requests.request(url=search_url, headers=headers, method="GET")
#
# res = req.text

# print(res)
# json_obj = json.loads(raw)
# body = json_obj['response']['song']
# lyrics_path = body['path']
lyrics_url = "https://genius.com/Taylor-swift-wildest-dreams-lyrics"

if True:
    req = requests.request(url=lyrics_url, headers=headers,method="GET")

    html = BeautifulSoup(req.text, "html.parser")
    [h.extract() for h in html('script')]
    lyrics = html.find("div", class_="lyrics").get_text()
    lyrics = lyrics.replace("\n", " ")
    lyrics = re.sub(r'\[.+?\]', "",lyrics)
    #TODO: regex to remove [Verse] etc
    songFile = open("train/swift/wildest_dreams.txt", "w")
    songFile.write(lyrics)
    songFile.close()

# print(raw)
# json_obj = json.loads(raw)
# body = json_obj
# print(body)
