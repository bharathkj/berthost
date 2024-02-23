import requests
import json

# Flask server URL
flask_url = 'http://192.168.29.182:5000/analyze-emotion'  # Update with your actual Flask server URL

# Text data to send for analysis
text_data = "pokemon is my least favorite show. i despise it"

# Prepare the request payload
payload = {'text_data': text_data}

# Send the HTTP POST request
try:
    response = requests.post(flask_url, json=payload)

    if response.status_code == 200:
        # Successfully received sentiment analysis results
        data = response.json()
        print('Sentiment Analysis Results:', data)
    else:
        # Handle error
        print('Error:', response.status_code)

except requests.exceptions.RequestException as e:
    # Handle network errors
    print('Error:', e)
