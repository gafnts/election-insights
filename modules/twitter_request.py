import os
import tweepy
import logging
import backoff
import requests
import pandas as pd


bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
client = tweepy.Client(bearer_token=bearer_token)


class TwitterRequest:
    def __init__(self, query: str, start_time: str, end_time: str, max_results: str) -> None:
        self.query = query
        self.start_time = start_time
        self.end_time = end_time
        self.max_results = max_results

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler("twitter_request.log")
        file_handler.setLevel(logging.INFO) 

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)  

        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
    
    @backoff.on_exception(
            backoff.expo, 
            (tweepy.errors.TooManyRequests, requests.exceptions.ReadTimeout),
            max_tries=5
        )
    def make_request(self) -> "TwitterRequest":
        self.query = f"{self.query} -is:retweet -is:reply"

        self.logger.info("Making request with query: %s", self.query)
        try:
            self.tweets = client.search_recent_tweets(
                query = self.query,
                start_time = self.start_time,
                end_time = self.end_time,
                max_results = self.max_results,
                tweet_fields = [
                    "id", "author_id", "created_at", "text", 
                    "public_metrics", "possibly_sensitive", "lang"
                ],
                user_fields = [
                    "id", "username", "name", "location", "created_at", "description", 
                    "profile_image_url", "verified", "public_metrics"
                ],
                expansions = [
                    "author_id", "referenced_tweets.id"
                ]
            )
            if self.tweets is None or self.tweets.data is None:
                self.logger.error("No tweets returned from request")
                return self
        except Exception as e: 
            self.logger.error("Exception occurred", exc_info=True)
            raise 

        self.logger.info("Request completed successfully.")
        return self

    def extract_tweets(self) -> "TwitterRequest":
        self.logger.info("Starting to extract tweets.")
        tweet_data = []
        for tweet in self.tweets.data:
            tweet_dict = {key: getattr(tweet, key) for key in tweet.data.keys()}
            public_metrics = tweet_dict.pop('public_metrics')
            tweet_dict.update(public_metrics)
            tweet_data.append(tweet_dict)

        self.df = pd.DataFrame(tweet_data)
        self.logger.info("Tweets extraction completed successfully.")
        return self

    def extract_users(self) -> "TwitterRequest":
        self.logger.info("Starting to extract users.")
        users = {user.id: user for user in self.tweets.includes['users']}
        for key, user in users.items():
            user_data = {f"user_{key}": getattr(user, key) for key in user.data.keys()}
            public_metrics_user = user_data.pop('user_public_metrics')
            user_data.update({f"user_{k}": v for k, v in public_metrics_user.items()})
            users[key] = user_data

        self.df['user_data'] = self.df['author_id'].apply(lambda x: users[x])

        user_columns = pd.json_normalize(self.df['user_data']).columns
        for col in user_columns:
            self.df[col] = self.df['user_data'].apply(lambda x: x.get(col, None))

        self.df = self.df.drop(columns = ['user_data'])
        self.logger.info("User data extraction completed successfully.")
        return self

    def segregate(self) -> "TwitterRequest":
        self.logger.info("Starting to segregate data.")
        self.tweets_df = self.df[[
            "id", "author_id", "created_at", "text", "possibly_sensitive", "retweet_count",
            "reply_count", "like_count", "quote_count", "impression_count", "lang"
        ]]

        self.users_df = self.df[[
            "user_id", "user_username", "user_name", "user_location", "user_created_at",
            "user_description", "user_profile_image_url", "user_verified",
            "user_followers_count", "user_following_count", "user_tweet_count", "user_listed_count"
        ]]
        self.logger.info("Data segregation completed successfully.")
        return self

    def preprocess(self, tweets_prefix: str, users_prefix: str) -> pd.DataFrame:
        self.logger.info("Starting to preprocess data.")
        self.tweets_df = (
            self.tweets_df
            .rename(columns={
            "id": f"{tweets_prefix}tweet",
            "author_id": f"{tweets_prefix}usuario",
            "created_at": f"{tweets_prefix}fecha",
            "text": f"{tweets_prefix}texto",
            "possibly_sensitive": f"{tweets_prefix}sensitivo",
            "retweet_count": f"{tweets_prefix}retweets",
            "reply_count": f"{tweets_prefix}replies",
            "like_count": f"{tweets_prefix}likes",
            "quote_count": f"{tweets_prefix}quotes",
            "impression_count": f"{tweets_prefix}impresiones",
            "lang": f"{tweets_prefix}idioma"
            })
            .assign(tw_fecha = lambda x: pd.to_datetime(x.tw_fecha).dt.date)
        )

        self.users_df = (
            self.users_df
            .rename(columns={
                "user_id": f"{users_prefix}usuario",
                "user_username": f"{users_prefix}handle",
                "user_name": f"{users_prefix}nombre",
                "user_location": f"{users_prefix}ubicacion",
                "user_created_at": f"{users_prefix}fecha_creacion",
                "user_description": f"{users_prefix}descripcion",
                "user_profile_image_url": f"{users_prefix}imagen",
                "user_verified": f"{users_prefix}verificado",
                "user_followers_count": f"{users_prefix}seguidores",
                "user_following_count": f"{users_prefix}siguiendo",
                "user_tweet_count": f"{users_prefix}tweets",
                "user_listed_count": f"{users_prefix}listas"
            })
            .assign(us_fecha_creacion = lambda x: pd.to_datetime(x.us_fecha_creacion).dt.date)
        )
        self.logger.info("Data preprocessing completed successfully.")
        return self.tweets_df, self.users_df