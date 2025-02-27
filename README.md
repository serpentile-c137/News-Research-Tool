# News-Research-Tool

News research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the given URLs.

## Features

* Load URLs or upload text files containing URLs to fetch article content.
* Process article content through LangChain's UnstructuredURL Loader
* Construct an embedding vector using OpenAI's embeddings and leverage FAISS, a powerful similarity search library, to enable swift and * Effective retrieval of relevant information
* Interact with the LLM's (Google Gemini) by inputting queries and receiving answers along with source URLs.

## Installation

1. Clone this repository to your local machine using:
```bash
git clone https://github.com/serpentile-c137/News-Research-Tool.git
cd News-Research-Tool
```

2. Install libraries
```bash
pip install -r requirements
```

3. 4.Set up your Gemini API key by creating a .env file in the project root and adding your API
```bash
GOOGLE_API_KEY=your_api_key
```

4. Run the Streamlit app
```bash
streamlit run main.py
```

#### Project Url : https://ai-news-research-tool.streamlit.app/

#### Output: 
![alt text](https://github.com/serpentile-c137/News-Research-Tool/blob/main/news-research-tool.webm)

https://github.com/user-attachments/assets/bcef8bd8-0904-4d9e-bca4-9c44a8643263


