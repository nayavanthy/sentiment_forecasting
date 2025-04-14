import requests

url = "https://780c-34-125-244-167.ngrok-free.app/predict"
payload = {
    "instances": [
        {"texts": ["I love this!", "It's okay", "I hate it"]}
    ]
}
response = requests.post(url, json=payload)
print(response.json())
