
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
