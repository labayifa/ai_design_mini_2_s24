from datetime import datetime

import boto3
from botocore.config import Config

my_config = Config(
    region_name = 'us-east-1'
)

# Get the service resource.

session = boto3.Session(
    aws_access_key_id='<Access key>',
    aws_secret_access_key='<Secret key>'
)

dynamodb = session.resource('dynamodb', config=my_config)
scores_table = dynamodb.Table('IrisExtended')
