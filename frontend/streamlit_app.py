import streamlit as st
import requests

API_URL = "https://interactive-chatbot-using-langchain.onrender.com/chat"

st.title("Human-in-the-Loop AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        response = requests.post(API_URL, json={"message": user_input, "session_id": "1"})
        try:
            bot_reply = response.json().get("response")
        except:
            bot_reply = "Server error"
        
        st.session_state.messages.append(("user", user_input))
        st.session_state.messages.append(("bot", bot_reply))

for sender, msg in st.session_state.messages:
    st.chat_message(sender).markdown(msg)

