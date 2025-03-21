# System imports
import os
import sys
import asyncio
from dotenv import load_dotenv  # Import dotenv
load_dotenv()   # Load environment variables from .env

# PandasAI imports
import streamlit as st
import pandas as pd
import pandasai as pai
from pandasai_openai import OpenAI

# Sandbox environment imports
from pandasai import Agent
from pandasai_docker import DockerSandbox
from guardrails.guardrails import execute_input_guardrails

# Lida imports
from lida import llm, Manager, TextGenerationConfig
from PIL import Image
import io
import base64


# --- Configuration ---
# Keys for storing session state and model configurations
CHAT_HISTORY_KEY = 'messages'  # Stores chat history in session state
MODEL_KEY = 'openai_model'  # Stores the selected OpenAI model
DATASETS_KEY = 'uploaded_datasets'  # Stores uploaded datasets
SELECTED_DATASET_KEY = 'selected_dataset'  # Stores the currently selected dataset


# --- OpenAI API Key Handling ---
# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

try:
    # Instantiate OpenAI client
    openai_llm = OpenAI(api_key=OPENAI_API_KEY)

    # Configure PandasAI to use the OpenAI LLM
    pai.config.set({"llm": openai_llm})

    # Configure Lida for chart visualisation
    lida = Manager(text_gen=llm("openai", api_key=OPENAI_API_KEY))
    textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)

    # Initialise Docker sandbox for secure execution of PandasAI queries
    sandbox = DockerSandbox()
    sandbox.start()
except Exception as e:
    # Handle initialisation errors and exit the application if necessary
    st.error(f"Error initialising OpenAI client: {e}", icon="🚨")
    sys.exit(f"Error initialising OpenAI client: {e}")


# -- Helper Functions --
def upload_datasets():
    """
    Displays the file uploader widget to allow users to upload multiple CSV or XLS files.
    Loads the uploaded datasets into the session state and returns them as a dictionary.

    Returns:
        datasets (dict): A dictionary where the keys are filenames and the values are DataFrames.
    """

    # Display file uploader widget to allow users to upload multiple CSV or XLS files
    uploaded_files = st.file_uploader("Upload datasets (CSV or XLS)", type=["csv", "xls"], accept_multiple_files=True)
    
    # Initialise an empty dictionary to store the datasets after loading
    datasets = {}

    # Get the list of filenames of the currently uploaded files
    uploaded_filenames = [uploaded_file.name for uploaded_file in uploaded_files]

    # If datasets already exist in session state, clean up files that were removed from the uploader
    if DATASETS_KEY in st.session_state:
        current_files = list(st.session_state[DATASETS_KEY].keys())
        for file in current_files:
            # Check if the current file is no longer part of the uploaded files list, and remove it from session state
            if file not in uploaded_filenames:
                del st.session_state[DATASETS_KEY][file]  
    
    # Check if any files have been uploaded
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                try:
                    # Check if the uploaded file is a CSV and read it into a DataFrame
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    # Check if the uploaded file is an XLS and read it into a DataFrame
                    elif uploaded_file.name.endswith('.xls'):
                            df = pd.read_excel(uploaded_file)
                    # If the file type is unsupported, display a warning message
                    else:
                        st.warning(f"Unsupported file type: {uploaded_file.name}. Please upload CSV or XLS.")
                        continue

                    # Add the loaded DataFrame to the datasets dictionary with the filename as the key
                    datasets[uploaded_file.name] = df
                    st.success(f"Dataset '{uploaded_file.name}' uploaded successfully.")
                except Exception as e:
                    st.error(f"Error reading file '{uploaded_file.name}': {e}")

    # Return uploaded datasets
    return datasets


def process_query(data, prompt: str):
    """
    Processes the user query using the provided dataset and prompt.

    Args:
        data (DataFrame): The dataset to process the query with.
        prompt (str): The user's input query.

    Returns:
        str: The response from processing the query with the dataset.
    """

    try:
        # Initialise an Agent object
        agent = Agent(data, sandbox=sandbox)
        # Use the agent to process the query and get a response based on the provided prompt
        response = agent.chat(prompt)
        return response
    except Exception as e:
        return f"Error processing query with PandasAI: {e}"

def visualisation_lida(data, prompt: str):
    """
    Generates a summary of the data and visualises it using the LIDA library.

    Args:
        data (DataFrame): The dataset to summarise and visualise.
        prompt (str): The user's prompt for the visualisation goal.

    Returns:
        chart: The first chart generated from the visualisation.
    """

    # Generate a summary of the data
    summary = lida.summarize(data, summary_method="default", textgen_config=textgen_config)
    # Visualise the summary
    charts = lida.visualize(summary=summary, goal=prompt, textgen_config=textgen_config)

    # If charts are generated, return the first chart from the list
    if charts:
        return charts[0]


def handle_next_query_dataset_selection():
    """
    Handles the selection of a different dataset for the next query.
    Provides options to show the top N rows of the selected dataset.

    Returns:
        None
    """

    # --- Dataset Selection for Next Query ---
    # Get the list of dataset names available in the session state after the current query
    dataset_names_after_query = list(st.session_state[DATASETS_KEY].keys())

    # If there are datasets available, provide the user with the option to select a dataset for the next query
    if dataset_names_after_query:
        selected_dataset_after_query = st.selectbox(
            "Select a (different) dataset to query (if needed)", 
            dataset_names_after_query, 
            key=f"dataset_selector_after_query_{len(st.session_state[CHAT_HISTORY_KEY])}", 
            index=None
        )
        
        # If the user selects a different dataset, update the session state with the selected dataset
        if selected_dataset_after_query != st.session_state.get(SELECTED_DATASET_KEY):
            st.session_state[SELECTED_DATASET_KEY] = selected_dataset_after_query
            st.write(f"Selected Dataset for next query: **{selected_dataset_after_query}**")
        else:
            st.write("No datasets available to select for the next query.")
    else:
        st.write("No datasets available to select for the next query.")

    # --- Resetting the checkbox for "Show Top N Rows" ---
    # Provide a checkbox for the user to decide if they want to view the top N rows of the selected dataset
    show_top_n = st.checkbox(
        "Show top N rows of the dataset?", 
        key=f"checkbox_{st.session_state.get(SELECTED_DATASET_KEY, '')}_{len(st.session_state[CHAT_HISTORY_KEY])}", 
        value=False)
    
    # If the user checks the box, display an input field to specify the number of rows to show
    if show_top_n:
        if selected_df is not None:

            # Validation checks
            # - Number cannot be smaller than 1
            # - Number cannot be larger than number of rows in dataset
            num_rows = selected_df.shape[0]
            n_value = st.number_input("Enter the number of rows (N):", min_value=1, value=1, step=1, max_value=num_rows)
            if n_value > 0:
                st.subheader(f"Top {n_value} Rows of '{selected_dataset}'")
                st.dataframe(selected_df.head(n_value))
            else:
                st.warning("Please select a dataset first to show top rows.")


async def main():
    # --- Dataset Upload ---
    if DATASETS_KEY not in st.session_state:
        st.session_state[DATASETS_KEY] = {}

    # Allow users to upload datasets and update the session state with the uploaded datasets
    uploaded_datasets = upload_datasets()
    st.session_state[DATASETS_KEY].update(uploaded_datasets)

    st.divider()  # Add a separator

    # Check if no datasets have been uploaded, and show a message if that's the case
    if DATASETS_KEY not in st.session_state or not st.session_state[DATASETS_KEY]:
        st.write("No datasets uploaded yet.")
        return

    # --- Initialise Chat History ---
    if CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[CHAT_HISTORY_KEY] = []

    # Display all chat messages stored in the session history
    for i, message in enumerate(st.session_state[CHAT_HISTORY_KEY]):
        with st.chat_message(message["role"]):
            if message.get("type") == "image":
                st.image(message["content"], caption=message.get("caption", ""))
            else:
                st.write(message["content"])

    # --- Dataset Selection for Query ---
    dataset_names = list(st.session_state[DATASETS_KEY].keys())

    if not dataset_names:
        st.write("Please upload a dataset first to query.")
        return

    selected_dataset = st.selectbox("Select a dataset to query", dataset_names, key="dataset_selector", index=None)
    
    # Store the selected dataset name in session state
    st.session_state[SELECTED_DATASET_KEY] = selected_dataset
    
    # Retrieve the dataframe for the selected dataset from session state
    selected_df = st.session_state[DATASETS_KEY].get(selected_dataset)

    # If a dataset is selected, display its name; otherwise, display a message indicating no dataset is selected
    if selected_df is not None:
        st.write(f"Selected Dataset: **{selected_dataset}**)")
    else:
        st.write("No dataset selected.")

    # --- "Show Top N Rows" Interaction ---
    # Display a checkbox for the user to decide whether to show the top N rows of the selected dataset
    show_top_n = st.checkbox("Show top N rows of the dataset?", key=f"checkbox_{st.session_state.get(SELECTED_DATASET_KEY, '')}", value=False)
    
    # If the checkbox is checked, prompt the user to input how many top rows they want to display
    if show_top_n:
        if selected_df is not None:
            # Validation checks
            num_rows = selected_df.shape[0]
            n_value = st.number_input("Enter the number of rows (N):", min_value=1, value=1, step=1, max_value=num_rows)
            # If the checkbox is checked, prompt the user to input how many top rows they want to display
            if n_value > 0:
                st.subheader(f"Top {n_value} Rows of '{selected_dataset}'")
                st.dataframe(selected_df.head(n_value))
            else:
                st.warning("Please select a dataset first to show top rows.")

    # Accept user input through a chat input box
    if prompt := st.chat_input("Ask a question about your data"):
        # Add user message to chat history
        st.session_state[CHAT_HISTORY_KEY].append({"role": "user", "content": prompt, "dataset": st.session_state.get(SELECTED_DATASET_KEY)})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if selected_df is not None:
                # Convert the selected dataframe into a PandasAI DataFrame to process queries
                selected_data = pai.DataFrame(selected_df)

                async def run_query():
                    # Execute input guardrails (e.g., validation or filtering) on the user input
                    guardrail_response = await execute_input_guardrails(prompt, sample_df=selected_data.head(5))
                    
                    # If the guardrails provide a response (e.g., validation fails), display the response and stop further execution
                    if guardrail_response:
                        st.markdown(guardrail_response)
                        return None 
                    
                    try:
                        response = process_query(selected_data, prompt)
                        return response
                    except Exception as e:
                        print(f"Error processing query with PandasAI: {e}")
                        return f"An error occurred while processing your query: {e}"

                response = await run_query()

                # If a valid response has been generated:
                # Display the response - we need to handle different response types
                if response != None:

                    # a) Check if the response is a chart generated by PandasAI
                    if isinstance(response, pai.core.response.chart.ChartResponse):     # Check if it's a matplotlib figure
                        # Generate a visualisation using Microsoft LIDA
                        visualisation = visualisation_lida(selected_df, prompt)
                        # If a valid visualisation is returned, display it
                        if visualisation and visualisation.raster:
                            imgdata = base64.b64decode(visualisation.raster)
                            img = Image.open(io.BytesIO(imgdata))
                            st.image(img, caption=prompt, use_container_width=True)
                            
                            # Add the image response to the chat history
                            st.session_state[CHAT_HISTORY_KEY].append({
                                "role": "assistant",
                                "type": "image",
                                "content": img,
                                "caption": prompt,
                                "dataset": selected_df}
                            )

                    # b) Check if the response is a dataframe generated by PandasAI
                    elif isinstance(response, pai.core.response.dataframe.DataFrameResponse):   # Check if it's a matplotlib figure
                        # Display it in the Streamlit app
                        st.write(response)
                        # Add the dataframe response to the chat history
                        st.session_state[CHAT_HISTORY_KEY].append({
                            "role": "assistant",
                            "type": "dataframe",
                            "content": response,
                            "dataset": selected_data}
                        )
                        
                        # Generate and display a visualisation using Microsoft LIDA
                        visualisation = visualisation_lida(selected_df, prompt)
                        if visualisation and visualisation.raster:
                            imgdata = base64.b64decode(visualisation.raster)
                            img = Image.open(io.BytesIO(imgdata))
                            st.image(img, caption=prompt, use_container_width=True)

                            # Add the image response to the chat history
                            st.session_state[CHAT_HISTORY_KEY].append({
                                "role": "assistant",
                                "type": "image",
                                "content": img,
                                "caption": prompt,
                                "dataset": selected_df
                            })

                    # If the response is a string / number
                    else:
                        # Display the response as markdown (text)
                        st.markdown(response)

                        # Add the text response to the chat history
                        st.session_state[CHAT_HISTORY_KEY].append({
                            "role": "assistant",
                            "type": "text",
                            "content": response,
                            "dataset": selected_data
                        })

            else:
                st.write("Error: Could not initialize PandasAI or no dataset selected.")

            st.divider() 

            # Handle the next query
            handle_next_query_dataset_selection() 

# Main entry point of the application
if __name__ == '__main__':
    try:
        asyncio.run(main())
    finally:
        # Ensure that the sandbox is stopped when the app finishes
        sandbox.stop()