from g4f.client import Client
from g4f.Provider import ChatgptAi,You,Liaobots,Bing,ChatForAi,Chatgpt4Online,ChatgptNext,ChatgptX,Gemini,GeminiPro,GptTalkRu,Chatgpt4Online,ChatgptNext,FlowGpt
import nest_asyncio
# nest_asyncio.apply()
from g4f.client import Client

client = Client(provider=You)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)