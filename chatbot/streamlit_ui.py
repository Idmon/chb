import streamlit as st
from streamlit_chat import message
from salesgpt.agents import SalesGPT

def streamlit_interface(agent):
    def conversational_chat(query):
        agent.human_step(query)
        agent.step()
        result = agent.conversation_history[-1]
        st.session_state['history'].append((query, result))
        return result
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything you like. 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]
        
    # Container for the chat history
    response_container = st.container()
    # Container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True): 
            user_input = st.text_input("Query:", placeholder="Type your message here :)", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', logo="https://customflagsaustralia.com.au/wp-content/uploads/2022/08/Aramean-Syriac-Flag-Old-Aramean-Syriac-Peoples.jpg")
                message(st.session_state["generated"][i], key=str(i))