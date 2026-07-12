from google import genai

client = genai.Client(api_key="AQ.Ab8RN6IjYH6_jWXCzfJc3HRllYXLmgSDdS-W66rm9jXD5s2uiw")
response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Say hello",
)
print(response.text)