import streamlit as st
from openai import OpenAI
import os

async def check_for_irrelevant_input(user_request, df):
    """
    Function to check if the user's question is related to the dataset uploaded by them.
    It communicates with OpenAI's GPT model to perform the assessment.

    Parameters:
    - user_request: str - The question or request made by the user.
    - df: pandas.DataFrame - The dataset that the user has uploaded.

    Returns:
    - None if the question is related to the dataset.
    - A warning message if the question is irrelevant to the dataset.
    """
    
    # Define the system prompt that will guide the AI in analysing the user's request
    system_prompt = """
    You are a helpful assistant designed to determine if a user's request is related to the data they have uploaded.

    Your goal is to assess if the request can be potentially addressed using the information in the provided datasets.

    Here are the column names in the uploaded dataset:
    {column_names}

    Your task is to analyse the user's request: "{user_question}".

    Please respond with "allowed" if the request appears to be related to the uploaded data, or "not_allowed" if it is clearly unrelated.
    """

    # Extract column names from the dataframe (if it exists) to be included in the prompt
    column_names = list(df.columns) if df is not None and not df.empty else "No data available yet."
    
    # Format the prompt with the user's question and the column names from the dataset
    prompt = system_prompt.format(user_question=user_request, column_names=column_names)

    try:
        # Initialise the OpenAI client with the API key
        openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))

        # Construct the message structure for the chat completion request
        messages = [
            {
                "role": "system",
                "content": "You are a helpful topic assessment assistant."
            },
            {
                "role": "user",
                "content": prompt
            },
        ]

        # Make the API call to OpenAI to assess the request
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=messages,  
            temperature=0.0 
        )

        # Extract and clean up the response from OpenAI
        guardrail_result = response.choices[0].message.content.strip().lower()

        # Determine if the user's request is allowed or not based on the response
        if guardrail_result == "allowed":
            return None  
        else:
            return "I'm sorry, your question appears to be unrelated to the data you have provided. Please ask a question specifically about the uploaded datasets."
    
    except Exception as e:
        st.warning(f"Error during topical guardrail check: {e}")
        return "An error occurred while checking if your question is related to the data."
