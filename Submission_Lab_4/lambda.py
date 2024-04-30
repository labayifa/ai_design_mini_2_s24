import json
import logging
import boto3

# Get the service resource.

client = boto3.client('dynamodb')


def lambda_handler(event, context):
    # TODO implement

    print(event)
    for rec in event['Records']:
        print(rec)
        if rec['eventName'] == 'INSERT':
            UpdateItem = rec['dynamodb']['NewImage']
            print(UpdateItem)

            # lab4 code goes here
            confidences = float(UpdateItem['Probability']['S'])
            if UpdateItem['Class']['S'] != UpdateItem['Actual']['S'] or confidences < 0.9:
                response = client.put_item(TableName='IrisExtendedRetrain', Item=UpdateItem)
                logging.info(f"Inserting logs for Retraining model with {UpdateItem} of probability {confidences}")
                print(response)
    return {
        'statusCode': 200,
        'body': json.dumps('IrisExtendedRetrain Lambda return')
    }
