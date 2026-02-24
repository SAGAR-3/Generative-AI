from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = OpenAI(model="gpt-3.5-turbo-0125", temperature=0.7)
response = llm.invoke("What is the capital of France?")
print(response)  # Expected output: "The capital of France is Paris."

