import urllib
import json
import random
import os
import pickle
import oauth2


# start getWeeksData
def getTwitterData(keyword):
    tweets = getData(keyword)

    # Write data to a pickle file
    filename = 'data-set/fetchedTweets/fetchedTweets_' + urllib.unquote(keyword.replace("+", " ")) + '_' + str(
        int(random.random() * 10000)) + '.txt'
    outfile = open(filename, 'wb')
    pickle.dump(tweets, outfile)
    outfile.close()
    return tweets


# end

def parse_config():
    config = {}
    # from file args
    if os.path.exists('config.json'):
        with open('config.json') as f:
            config.update(json.load(f))
    else:
        print("Error : Config File not Found !!")
    # should have something now
    return config


def oauth_req(url):
    config = parse_config()
    consumer = oauth2.Consumer(key=config.get('consumer_key'), secret=config.get('consumer_secret'))
    token = oauth2.Token(key=config.get('access_token'), secret=config.get('access_token_secret'))
    client = oauth2.Client(consumer, token)

    resp, content = client.request(
        url,
        method="GET",
        body=None or '',
        headers=None
    )
    return content


# start getTwitterData
def getData(keyword, maxt_tweets=50):
    url = 'https://api.twitter.com/1.1/search/tweets.json?'
    data = {'q': keyword, 'lang': 'en', 'result_type': 'recent', 'count': maxt_tweets, 'include_entities': 0}

    url += urllib.urlencode(data)

    response = oauth_req(url)
    json_data = json.loads(response)
    tweets = []
    if 'errors' in json_data:
        print("API Error")
        print(json_data['errors'])
    else:
        for item in json_data['statuses']:
            tweets.append(item['text'])
    return tweets
    # end
