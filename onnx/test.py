import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# Replace with your actual Lambda function URL
request = {
    "url": 'https://github.com/jg5xraydelta/SkinCancerCapstone/blob/main/data/test/malignant/1.jpg?raw=true'
}

result = requests.post(url, json=request).json()
print(result)