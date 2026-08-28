import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("WATSONX_API_KEY")

# الخطوة 1: احصل على Token
token_url = "https://iam.cloud.ibm.com/identity/token"

response = requests.post(token_url, data={
    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
    "apikey": API_KEY
})

print("Status:", response.status_code)

if response.status_code == 200:
    token = response.json()["access_token"]
    print("✅ Token received successfully!")
    print("Token starts with:", token[:20], "...")
else:
    print("❌ Error:", response.text)