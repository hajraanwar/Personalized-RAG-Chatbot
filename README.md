# AI-Powered Chatbot with Llama and OpenAI

## Overview
This is a Streamlit-based AI-powered chatbot that integrates **Meta-Llama-3-8B-Instruct** for local response generation and **OpenAI's GPT-3.5 Turbo** for text embedding and summarization. The chatbot allows users to upload `.txt` files, process and store them in a ChromaDB vector database, retrieve relevant documents, and generate AI-driven responses.

## Features
- **Llama Model Integration:** Uses Meta-Llama-3-8B-Instruct for response generation.
- **Chroma Vector Database:** Stores and retrieves document chunks efficiently.
- **OpenAI Embeddings:** Utilizes `text-embedding-3-small` for document indexing.
- **Token Management:** Counts and manages tokens using `tiktoken` to prevent exceeding model limits.
- **Chat History:** Maintains past conversations for better user experience.
- **Retry Mechanism:** Implements backoff retrying for handling API rate limits.

## Technologies Used
- Python
- Streamlit
- LangChain
- OpenAI API
- ChromaDB
- Transformers (Hugging Face)
- Tiktoken
- PyTorch

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip

### Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Up ChromaDB Storage
Create a directory for persistent storage:
```bash
mkdir chroma_storage
```

### Environment Variables
Set up the required API keys in your environment:
```bash
export OPENAI_API_KEY='your_openai_api_key'
```

## Usage
### Running the Application
```bash
streamlit run app.py
```

### Uploading a File
1. Click on the file uploader and select a `.txt` file.
2. The system will split and store document chunks in ChromaDB.
3. A confirmation message will indicate successful storage.

### Querying the Chatbot
1. Enter a query in the text input field.
2. Click **Submit** to retrieve relevant document chunks and generate a response using the Llama model.
3. The chatbot will respond and append the interaction to chat history.

## Project Structure
```
├── app.py                 # Main application script
├── requirements.txt       # Required dependencies
├── chroma_storage/        # Persistent storage for ChromaDB
├── README.md              # Project documentation
```

## Known Issues & Limitations
- **Llama model requires GPU:** Ensure your machine has sufficient GPU memory.
- **Token limitations:** Summarization is triggered when token limits exceed 15,000.
- **ChromaDB persistence:** Files are stored locally; cloud support can be added.

## Future Improvements
- Add a UI enhancement for better UX.
- Implement fine-tuning for Llama responses.
- Extend support for additional document formats (PDF, DOCX).


## License
This project is licensed under the MIT License.

