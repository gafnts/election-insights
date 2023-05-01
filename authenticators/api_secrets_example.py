
class TwitterAuthenticator():
    def __init__(self):
        self.bearer_token = 'bearer_token'
        self.api_key = 'api_key'
        self.api_key_secret = 'api_key_secret'
        self.access_token = 'access_token'
        self.access_token_secret = 'access_token_secret'

class RDSAuthenticator():
    def __init__(self):
        self.host = 'host'
        self.port = 8080
        self.database = 'database'
        self.user = 'user'
        self.password = 'password'

class OpenAIAuthenticator():
    def __init__(self):
        self.api_key = 'api_key'