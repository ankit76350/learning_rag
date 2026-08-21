This will give the account number 
python -c "from dotenv import load_dotenv; load_dotenv(); import boto3; print(boto3.client('sts').get_caller_identity()['Account'])"

this will load all the .env file
load_dotenv()


i run this beause i have run the ingestion pipline twice that created the duplicate vector
Remove-Item -Recurse -Force db\chroma_db
python ingestion_pipeline.py


END