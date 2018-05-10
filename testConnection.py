import get_twitter_data

keyword = 'tamam'
time = 'today'
twitterData = get_twitter_data.TwitterData()
tweets = twitterData.getTwitterData(keyword, time)
print(tweets)
