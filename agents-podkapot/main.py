from loop import Chat
from agents import *

if __name__ == "__main__":
    bot = Chat(Router=Router, Recurser=Recurser, memory_limit=10)

    while True:
        user_input = input("User: ")
        #print("User:", user_input)
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat.")
            break
        response = bot(user_input)