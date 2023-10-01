import os
import boto3


from dotenv import load_dotenv
load_dotenv()

# Replace these with your AWS credentials and S3 bucket name
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME')

# Choosing the correct file system path
FS_PATH = ""
IS_LOCAL = os.environ.get('IS_LOCAL', False)
if not IS_LOCAL:
    FS_PATH = os.environ.get('AWS_EFS_PATH') + "/"
