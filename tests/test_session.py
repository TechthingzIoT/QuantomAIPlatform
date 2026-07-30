from runtime.chat.session import ChatSession
from runtime.inference.engine import LlamaEngine
from runtime.config.settings import settings

print("Loading engine...")

engine = LlamaEngine(settings.model_path)

session = ChatSession(
    engine=engine,
    system_prompt="You are QAIR, an AI engineering assistant."
)

print("\n===== Conversation =====\n")

reply = session.ask("What is an ESP32?")
print(reply)

reply = session.ask("What did I just ask you?")
print(reply)

print("\n===== Session Stats =====")
print(session.stats())

print("\n===== History =====")

for message in session.history():
    print(message)
