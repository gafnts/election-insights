import os
import re
import json
import openai
import backoff
import requests
from modules import setup_logger


api_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = api_key


class OpenAIRequest:
    """
    This class is responsible for sending a request to the OpenAI API, in its
    GPT-3.5-turbo language model, to extract features from a given tweet.

    Every method (except the last one) returns the class instance, so that
    the methods can be chained together. The tweet is passed as a parameter,
    preprocessed and then sent to the model with a prompt that defines the
    features to be extracted.

    The class uses the backoff library to retry the request if it fails and the
    response is returned as a dictionary with the feature names as keys.
    """
    
    def __init__(self, tweet: str) -> None:
        self.tweet = tweet
        self.logger = setup_logger(__name__, "logs/openai_request.log")
    
    @staticmethod
    def remove_emojis_and_links(tweet: str) -> str:
        """
        This static method defines regular expressions to find and remove the emojis and 
        links in a given string.
        """

        if not isinstance(tweet, str):
            return ""

        # Remove URLs.
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        tweet = url_pattern.sub('', tweet)

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
        tweet = emoji_pattern.sub('', tweet)

        # Remove newline characters.
        tweet = tweet.replace('\n', ' ')
        return tweet

    def preprocess_text(self) -> "OpenAIRequest":
        """
        This method preprocesses the tweet before sending it to the OpenAI API.
        """
        self.tweet = OpenAIRequest.remove_emojis_and_links(self.tweet)
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
    
    def extract_features(self, prefix: str) -> dict:
        """
        This method defines the prompt that will be sent to the OpenAI API, makes the request
        and returns the response as a dictionary with the feature names as keys.
        """

        prompt = f"""
        El siguiente es un tweet que menciona a un candidato presidencial dentro de la contienda electoral 2023 en Guatemala. 
        Por favor, clasifícalo de acuerdo a las siguientes categorías:

        Valencia (sentimiento general): [positivo, negativo, neutro, otro]
        Emoción (emoción principal expresada): [felicidad, tristeza, enojo, miedo, sorpresa, disgusto, otro]
        Postura (actitud hacia el tema): [aprobación, desaprobación, esperanza, desilusión, indiferencia, confianza, desconfianza, otro]
        Tono (forma de expresarse): [agresivo, pasivo, asertivo, escéptico, irónico, humorístico, informativo, serio, inspirador, otro]

        Además, evalúalo utilizando una escala continua con rango de 0 a 1 en las siguientes dimensiones:

        Amabilidad (nivel de cortesía): [0.0 - 1.0]
        Legibilidad (facilidad de lectura): [0.0 - 1.0]
        Controversialidad (potencial para generar desacuerdo): [0.0 - 1.0]
        Informatividad (cantidad de información relevante y fundamentada): [0.0 - 1.0]

        Formatea tu respuesta como un diccionario de Python con las siguientes llaves:

        [{prefix}valencia, {prefix}emocion, {prefix}postura, {prefix}tono, {prefix}amabilidad, {prefix}legibilidad, {prefix}controversialidad, {prefix}informatividad]

        Tweet: '''{self.tweet}'''
        """
        
        try:
            response = OpenAIRequest.make_request(prompt)
            if response is None:
                self.logger.error("Received invalid response from OpenAI")
                return None
            response = json.loads(response)

        except Exception as e:
            self.logger.error(f"Exception during API request: {e}")
            return None

        return response