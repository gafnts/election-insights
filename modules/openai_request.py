import os
import re
import time
import json
import openai
import backoff
import requests
import pandas as pd
from modules import setup_logger


api_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = api_key


class ExtractFeatures:
    def __init__(self, tweets: pd.DataFrame) -> None:

        # Initialize parameters.
        self.tweets = tweets

        # Initialize logger.
        self.logger = setup_logger(__name__, "logs/openai_request.log")
    
    @staticmethod
    def remove_emojis_and_links(text: str) -> str:
        """
        This static method defines regular expressions to find and remove the emojis and 
        links in a given string.
        """

        # Remove URLs.
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = url_pattern.sub('', text)

        # Remove emojis.
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # Emoticons.
            u"\U0001F300-\U0001F5FF"  # Symbols & pictographs.
            u"\U0001F680-\U0001F6FF"  # Transport & map symbols.
            u"\U0001F1E0-\U0001F1FF"  # Flags (iOS).
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub('', text)

        # Remove newline characters.
        text = text.replace('\n', ' ')
        return text

    def preprocess_text(self) -> "ExtractFeatures":
        """
        This method preprocesses the tweets before sending them to the OpenAI API.
        """
        
        self.preprocessed_tweets = (
            self.tweets
            .assign(
                tw_texto = lambda x: x['tw_texto'].apply(lambda x: ExtractFeatures.remove_emojis_and_links(x))
            )
        )
        return self
    
    @staticmethod
    @backoff.on_exception(
        backoff.expo, 
        (openai.error.RateLimitError, requests.exceptions.ReadTimeout),
        max_tries=5
    )
    def make_request(prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0) -> str: 
        """
        This static method let us make the request to the OpenAI API. The temperature parameter 
        of the language model is set to 0 to ensure that the response is as deterministic as possible.
        """

        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, 
        )
        return response.choices[0].message["content"]
    
    def extract_features(self, prefix: str) -> pd.DataFrame:
        """
        This method extracts the features from the tweets using the GPT-3.5-turbo language model.
        
        It iterates over the tweets and for each one it sents a prompt to the OpenAI API.
        The response is a JSON object that is parsed and converted to a pd.DataFrame.

        This DataFrame is then concatenated to the original DataFrame and returned.
        """

        self.logger.info("The feature extraction process has started.")

        collector = []
        for _, row in self.preprocessed_tweets.iterrows():
            prompt = f"""
                El siguiente es un tweet que menciona a un candidato presidencial dentro de la contienda electoral 2023 en Guatemala. 
                Por favor, clasifícalo de acuerdo a las siguientes categorías:

                Valencia (sentimiento general): [positivo, negativo, neutro]
                Emoción (emoción principal expresada): [felicidad, tristeza, enojo, miedo, sorpresa, disgusto]
                Postura (actitud hacia el tema): [aprobación, desaprobación, esperanza, desilusión, indiferencia, confianza, desconfianza]
                Tono (forma de expresarse): [agresivo, pasivo, asertivo, escéptico, irónico, humorístico, informativo, serio, inspirador]

                Además, evalúalo utilizando una escala continua con rango de 0 a 1 en las siguientes dimensiones:

                Amabilidad (nivel de cortesía): [0.0 - 1.0]
                Legibilidad (facilidad de lectura): [0.0 - 1.0]
                Controversialidad (potencial para generar desacuerdo): [0.0 - 1.0]
                Informatividad (cantidad de información relevante y fundamentada): [0.0 - 1.0]

                Formatea tu respuesta como un diccionario de Python con las siguientes llaves:

                [{prefix}valencia, {prefix}emocion, {prefix}postura, {prefix}tono, {prefix}amabilidad, {prefix}legibilidad, {prefix}controversialidad, {prefix}informatividad]

                Tweet: '''{row['tw_texto']}'''
                """
            
            try:
                self.logger.error(f"Starting API request for tweet: {row['tw_texto']}")
                response = ExtractFeatures.make_request(prompt)
                self.logger.info(f"API response: {response}")
                response = json.loads(response)
                response = pd.DataFrame([response])
            except Exception as e:
                self.logger.error(f"Exception during API request: {e}")
                response = pd.DataFrame()
                
            collector.append(response)
            time.sleep(3)

        new_features = pd.concat(collector, axis=0, ignore_index=True)
        self.expanded_tweets = pd.concat([self.preprocessed_tweets, new_features], axis=1)

        self.logger.info("The feature extraction process has finished.")
        return self.expanded_tweets