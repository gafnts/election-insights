from authenticators import RDSAuthenticator

rds = RDSAuthenticator()
connection = psycopg2.connect(
    host = rds.host,
    port = rds.port,
    database= rds.database,
    user = rds.user,
    password = rds.password
)
connection.autocommit = True
cursor = connection.cursor()