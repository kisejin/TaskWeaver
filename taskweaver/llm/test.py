from g4f.client import Client
from g4f.Provider import ChatgptAi,You,Liaobots,Bing,ChatForAi,Chatgpt4Online,ChatgptNext,ChatgptX,Gemini,GeminiPro,GptTalkRu,Chatgpt4Online,ChatgptNext,FlowGpt
from g4f import get_model_and_provider

model, provider = get_model_and_provider("gpt-4-turbo", "Bing", stream=True)
import nest_asyncio
nest_asyncio.apply()
from g4f.client import Client

client = Client(provider=provider)
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "Hello, what model are you using ?"}],
    stream=True
)
for res in response:
    print(res.choices[0].delta.content)

response.close()