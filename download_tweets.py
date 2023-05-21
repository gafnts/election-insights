import os
import pandas as pd
from typing import Tuple
from modules import setup_logger
from modules import GetTweets
from datetime import datetime, timedelta


# Requests parameters.
candidates = [
    'sandra torres', 'zury ríos'
]

start_date = datetime(2023, 5, 15, 00, 00)
end_date = datetime(2023, 5, 17, 00, 00)
max_results = 10
tweets_prefix = 'tw_'
users_prefix = 'us_'


class DownloadTweets:
    def __init__(
            self, 
            candidates: list[str], 
            start_date: datetime,
            end_date: datetime, 
            max_results: int,
            tweets_prefix: str, 
            users_prefix: str
        ) -> None:

        # Initialize parameters.
        self.candidates = candidates
        self.start_date = start_date
        self.end_date = end_date
        self.max_results = max_results
        self.tweets_prefix = tweets_prefix
        self.users_prefix = users_prefix

        # Initialize logger.
        self.logger = setup_logger(__name__, "download_tweets.log")
        self.logger.info("DownloadTweets initialized.")

    def generate_dates(self) -> "DownloadTweets":
        """
        This method generates a list of date pairs, representing each day from the 
        defined start_date to the defined end_date.
        """

        # Generate date pairs.
        delta = timedelta(days=1)
        date = self.start_date
        
        self.dates = []
        while date < self.end_date:
            next_date = date + delta
            self.dates.append(
                (date.isoformat() + "Z", next_date.isoformat() + "Z")
            )
            date = next_date
        return self

    def get_batch(
            self, candidate: str, start_date: datetime, end_date: datetime
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This method downloads a batch of tweets and users for a given candidate and date 
        pair. It returns a tuple of two pd.DataFrames, one for tweets and one for users.
        """

        tweets, users = (
            GetTweets(
                query=candidate,
                start_time=start_date,
                end_time=end_date,
                max_results=self.max_results
            )
            .make_request()
            .tweets_to_dataframe()
            .users_to_dataframe()
            .segregate_dataframe()
            .preprocess_data(
                tweets_prefix=self.tweets_prefix,
                users_prefix=self.users_prefix
            )
        )

        # Add a column for the candidate mentioned in each tweet.
        tweets['candidato'] = candidate

        self.logger.info(f"Downloaded batch of {len(tweets)} tweets for candidate {candidate}.")
        return tweets, users
    
    def download_tweets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This method downloads all tweets and users for the defined candidates and dates.
        It returns a tuple of two pd.DataFrames, one for tweets and one for users.
        """
        
        self.generate_dates()
        self.logger.info(f"Generated {len(self.dates)} date pairs for tweet downloads.")

        # Collect tweets and users for each candidate.
        tweets_collector, users_collector = [], []
        for candidate in self.candidates:

            # Collect tweets and users for each date.
            dates_tweets_collector, dates_users_collector = [], []
            for start_date, end_date in self.dates:

                tweets, users = self.get_batch(candidate, start_date, end_date)
                dates_tweets_collector.append(tweets)
                dates_users_collector.append(users)

            tweets_collector.append(pd.concat(dates_tweets_collector))
            users_collector.append(pd.concat(dates_users_collector))

        self.tweets = pd.concat(tweets_collector, axis=0, ignore_index=True)
        self.users = pd.concat(users_collector, axis=0, ignore_index=True)

        self.logger.info(f"Downloaded a total of {len(self.tweets)} tweets and {len(self.users)} users.")
        return self.tweets, self.users


def main() -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from_twitter = DownloadTweets(
            candidates=candidates,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            tweets_prefix=tweets_prefix,
            users_prefix=users_prefix
        )

        tweets, users = from_twitter.download_tweets()
        return tweets, users
    
    except Exception as e:
        logger = setup_logger(__name__, "download_tweets.log")
        logger.error(f"Failed to download tweets: {e}")
        raise e


if __name__ == "__main__":

    # Download tweets.
    tweets, users = main()
    
    # Create data folder if it does not exist.
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save the data.
    tweets.to_csv('data/tweets.csv', index=False)
    users.to_csv('data/users.csv', index=False)

    logger = setup_logger(__name__, "download_tweets.log")
    logger.info("Data saved to 'data/tweets.csv' and 'data/users.csv'")
