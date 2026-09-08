DSA Instructor RAG System
A complete Retrieval-Augmented Generation (RAG) system built in Node.js that serves as an interactive Data Structures and Algorithms (DSA) expert tutor. It indexes local PDF notes (dsa.pdf) into a Pinecone vector database using LangChain and leverages Google Gemini to answer technical questions with grounded, context-aware explanations.

Architecture Overview
The system operates across two main phases:

Indexing Phase (index.js): Loads the source PDF, splits text into optimal chunks, generates dense vector embeddings via Google Generative AI, and indexes them in Pinecone.

Querying Phase (query.js): Accepts interactive terminal questions, reframes follow-up queries using chat history context, performs vector similarity search, constructs an augmented prompt, and queries Gemini under strict persona constraints.

Prerequisites & Dependencies
Node.js installed locally

Google Gemini API Key

Pinecone Account & Index

Dependencies
Install the required packages using npm:

Bash
npm install @langchain/pinecone @langchain/core @pinecone-database/pinecone @langchain/community @google/genai @langchain/google-genai @langchain/textsplitters dotenv pdf-parse readline-sync
Configuration
Create a .env file in the root directory and configure your credentials:

Code snippet
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=your_pinecone_index_name_here
Project Structure
Ensure your file directory matches the following layout:

Plaintext
├── dsa.pdf       # Source DSA reference document / textbook
├── index.js      # Document parsing, chunking, and vector insertion pipeline
└── query.js      # Interactive CLI query engine with conversational history & query rewriting
Usage Guide
Step 1: Index the Knowledge Base
Run the indexing script to parse dsa.pdf, break it into chunks, embed the contents, and upload the vectors to Pinecone:

Bash
node index.js
Step 2: Run the Interactive DSA Instructor
Start the terminal interface to chat with your RAG-powered instructor:

Bash
node query.js
Once running, ask any DSA-related question directly in your terminal:

Plaintext
Ask me anything--> Explain the time complexity of Merge Sort
