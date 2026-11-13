
# https://github.com/Tem1rtasova/ista131-hw6

"""
Name: Aiganym Temirtassova
Section Leader: Senait Gezaei
Date: November 13, 2026
Course: ISTA 131 - Homework 6

Description:
Analyze tweet sentiment using TextBlob,
summarize results with pandas, and create
visualizations with matplotlib.

Assistance: ChatGPT for coding guidance.
"""

import json
import statistics

import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob


def get_sentiment(filename):
    """
    Calculate the sentiment of tweets in a JSON file.

    Parameters:
        filename (str): Path to the JSON file.

    Returns:
        list: Mean polarity and sample standard deviation.
    """
    with open(filename, "r", encoding="utf-8") as file:
        tweets = json.load(file)

    polarities = []

    for tweet in tweets:
        sentiment = TextBlob(tweet).sentiment

        if sentiment.polarity == 0.0 and sentiment.subjectivity == 0.0:
            continue

        polarities.append(sentiment.polarity)

    if not polarities:
        return [0.0, 0.0]

    mean = statistics.mean(polarities)

    if len(polarities) < 2:
        return [mean, 0.0]

    std = statistics.stdev(polarities)

    return [mean, std]



def get_ct_sentiment_frame():
    """
    Create a DataFrame of tweet sentiment statistics.

    Parameters:
        None

    Returns:
        pandas.DataFrame: Mean and sample standard
        deviation of sentiment for Trump and Biden
        on October 5 and October 6.
    """
    trump_5 = get_sentiment("data/trump_231005_730am.json")
    trump_6 = get_sentiment("data/trump_231006_330pm.json")

    biden_5 = get_sentiment("data/biden_231005_730am.json")
    biden_6 = get_sentiment("data/biden_231006_330pm.json")

    data = {
        "10/5 mean": [trump_5[0], biden_5[0]],
        "10/5 std": [trump_5[1], biden_5[1]],
        "10/6 mean": [trump_6[0], biden_6[0]],
        "10/6 std": [trump_6[1], biden_6[1]]
    }

    frame = pd.DataFrame(data, index=["Trump", "Biden"])

    return frame
