import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

def chat_with_ai():
    print("--- AI Chatbot (Type 'quit' to exit) ---")

    # This list keeps track of the conversation history
    # We start with a "System" message to give the AI a personality.
    conversation_history = [
        {"role": "system", "content": "Yor are a sarcastic tech support bot."}
    ]

    while True:
        #user input
        user_input = input("\nYou: ")

        if user_input.lower() in ["quit", "exit"]:
            break

        # Adding user history
        conversation_history.append({"role": "user", "content": user_input})

        try:
            #api call

            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=conversation_history,
                temperature=0.7
            )


            ai_response = completion.choices[0].message.content
            print(f"AI: {ai_response}")


            conversation_history.append({"role": "assistant", "content": ai_response})

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_with_ai()
