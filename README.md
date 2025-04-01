DS 4300 - Spring 2025 Practical 02: Local Retrieval-Augmented Generation System Overview

This project involves building a local Retrieval-Augmented Generation (RAG) system for querying course notes from our Spring 2025 DS4300 class. The system ingests a collection of documents that contain the material from our class, indexes them using embedding models and vector databases, and then answers user queries by generating responses using a locally running LLM.

The following tools and techniques were used to implement this system:
Python for building the pipeline
Ollama and Mistral for running local LLMs
Vector Databases: Redis Vector DB, Chroma, and Pinecone
Embedding Models: Sentence Transformers, nomic-embed, and other models

Team Members:
Sofia Zorich
Julian Getsey
Olivia Ericsson
Augusta Crow

Project Structure
The project is organized into the following key files and directories:
/project-root
├── /redis_search.py           # File for searching Redis Vector DB
├── /chroma_search.py          # File for searching Chroma Vector DB
├── /pinecone_search.py        # File for searching Pinecone Vector DB
├── /redis_ingest.py           # File for ingesting documents into Redis Vector DB
├── /chroma_ingest.py          # File for ingesting documents into Chroma Vector DB
├── /pinecone_ingest.py        # File for ingesting documents into Pinecone Vector DB
├── /pinecone_class.py   # File for containing class for ingesting and searching using Pinecone
├── /data/                     # Directory containing all the documents from the class
│   └── /class_notes.pdf       # PDFs that contain the information from class
├── /testing/                  # Directory containing test files
│   └── /testing.py      # File that conducts performance tests on different combinations
│   └── /test_data.csv      # CSV file with test results and performance metrics
├── /README.md                 # This README file

Key Files
/redis_ingest.py, /chroma_ingest.py, /pinecone_ingest.py
 These files handle the ingestion of documents into the respective vector databases. They index the documents by extracting embeddings and storing them in the vector database for efficient querying.
 
/redis_search.py, /chroma_search.py, /pinecone_search.py
 These files contain functions for performing queries on the respective vector databases (Redis, Chroma, Pinecone). Each search function takes user input, retrieves relevant context from the vector database, and passes it to a locally running LLM to generate a response.
 
/data/
 This directory contains the documents (PDFs, text files, etc.) that represent the course notes, presentations, and articles. These documents are ingested into the vector databases for indexing.
 
/testing/
 This directory contains files related to the testing and evaluation of the system.

test_results.csv contains the results of the tests, including metrics like response accuracy, query time, and system resource usage.

Requirements:
Before running the project, ensure you have the following dependencies installed:
Python 3.8+
Ollama (for running local LLMs)
Redis (for Redis Vector DB, need a container in Docker to use)
Os/fitz (for ingesting)
Chroma (for Chroma Vector DB)
Pinecone (for Pinecone Vector DB, need an API key from website)
Sentence Transformers (for embedding models)

How to Run the Project:
Ingest Documents into Vector Databases
 Choose one of the ingest_* scripts to ingest documents into the corresponding vector database. For example, to ingest documents into Redis: 
 python ingest_redis.py
 
Search in Vector Databases
 Use one of the search_* scripts to search the vector database. For example, to search in Redis:
 python search_redis.py
 
This will return a relevant context from the indexed documents and generate a response based on the user’s query.

Test the System
 You can test the system by running the testing script with the test queries and checking the results:
 python test_system.py
 
This will run a set of pre-defined queries and store the results in test_results.csv. From there, you can tailor your ingest and search code in order to generate the best and quickest answers from the LLM. 
