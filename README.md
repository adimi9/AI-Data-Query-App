# AI-Powered Data Query App 📊

## Overview
This app allows users to upload CSV/Excel files and interact with them through AI-powered queries and visualisations. The app supports multiple datasets, lets you explore the data by viewing top N rows, and provides insights through questions and answers powered by PandasAI with OpenAI. Visualisations are handled using Microsoft LIDA.

---

## Main Features 🎉

### 1. **Multiple Dataset Upload 📥**
Users can upload multiple CSV or Excel files. The app uses Pandas to load and process the data.

### 2. **View Top N Rows 👀**
Select which dataset to view and define how many top rows to display. The number of rows is adjustable via a Streamlit slider, and Pandas handles the data display.

### 3. **Ask Questions & Get Answers 💬**
Once the data is uploaded, users can ask questions, which are answered using PandasAI powered by OpenAI. For visualisations, Microsoft LIDA is used to generate interactive charts. Users can easily query any dataset via a dropdown menu.

### 4. **Guardrails for Safe & Relevant Queries 🚧**
The app includes built-in guardrails to ensure safe and relevant interactions. These include:
- **Jailbreak Guardrail**: Prevents any attempts to bypass the system’s rules.
- **Malicious Input Detection**: Blocks potentially harmful content like SQL injections and XSS.
- **Irrelevant Input Check**: Ensures queries are related to the uploaded datasets.

These guardrails are implemented using Python, regular expressions, and PandasAI for query analysis.

### 5. **History of Prompts 🔄**
Users can view and reuse previous queries. The app tracks this history using Streamlit's session state, allowing for easy reference and re-use of past prompts.

---

## Tech Stack 💻
- **Pandas**: Used for reading and processing uploaded datasets.
- **Streamlit**: Powers the app interface, including file uploads, data display, dropdowns, and session state for history management.
- **PandasAI (OpenAI)**: Handles AI-powered question answering.
- **Microsoft LIDA**: Generates interactive visualisations.
- **Python**: Handles the backend logic, security features, and input filtering.

---

## Safety Considerations 🛡️

### 1. **File Size Limit (10MB)**
To ensure smooth performance and prevent system overload, the app enforces a file size limit of 10MB for any dataset uploaded. This measure ensures that the system can process the data efficiently without any performance degradation, particularly when dealing with large datasets that may be resource-intensive. Files exceeding the 10MB limit are automatically rejected, ensuring the app remains responsive and stable.

### 2. **Sandbox Environment with PandaAI 🔒**
The app uses PandaAI’s sandbox environment to run Python code generated by LLMs for user queries. This ensures that the code execution is isolated from the main system, preventing any security risks. The sandbox operates offline, has strict resource limits, and keeps the file system secure from any potential harmful actions.

### 3. **Guardrails Implemented 🚧**
To further enhance the app’s security and ensure safe user interactions, three key guardrails have been implemented:

#### a. **Jailbreak Prevention 🚫**
The app incorporates a jailbreak detection mechanism that prevents users from attempting to bypass the system’s safety protocols. 

**How it works**:
- The app looks for specific phrases like “ignore all previous instructions” or “do not follow any of my instructions.”
- When a match is detected, the request is rejected with a message informing the user of the security violation.

#### b. **Malicious Input Detection 🛑**
To protect against potentially harmful inputs such as SQL injection, XSS attacks, or other forms of malicious code, the app implements an input vaLIDAtion system. 

**How it works**:
- The app checks for typical malicious patterns, such as:
  - **SQL Injection**: Detects attempts to manipulate database queries using keywords like "drop table", "select *", or "insert into".
  - **XSS and Script Injection**: Identifies attempts to execute harmful code, such as JavaScript embedded in the input (e.g., `<script>`, `eval()`).
- If any of these patterns are detected, the app immediately blocks the input and informs the user of the potential threat.

#### c. **Irrelevant Input Detection ⚠️**
To maintain the focus and relevance of user queries, the app checks if the user’s input pertains to the uploaded dataset. 

**How it works**:
- The app uses a combination of predefined rules and OpenAI’s GPT models to analyse whether the user’s query is related to the uploaded data.
- If the question is unrelated, the app alerts the user that the query doesn’t match the dataset, helping users stay on track and avoid confusion.

---

## **Implementation 🚀**
#### **1. Clone the Repository**  
```sh
git clone https://github.com/your-username/ai-data-query-app.git  
cd ai-data-query-app
```

#### **2. Install Dependencies**  
Ensure you have Python installed, then run:
```sh
pip install -r requirements.txt  
```

#### **3. Set Up Environment Variables** 
Create a .env file in the project root and add your OpenAI API key:
```sh
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env  
```

#### **4. Set Up Environment Variables** 
Start the application with:
```sh
streamlit run app.py
```






