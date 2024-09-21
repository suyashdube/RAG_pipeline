# RAG_pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline to process resumes, retrieve relevant information based on user queries, and generate context-aware responses. The system leverages a combination of vector-based retrieval using embeddings and language models for response generation. The entire pipeline is integrated into a REST API for querying and testing purposes.

Resume data taken from: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset?resource=download

**Codebase**
- given in ipynb format for better understanding of the code
- to test the working of model, query can be passed to the rag pipeline and output can be generated. A working example is shown in the notebook.
- ipynb notebook can be downloaded and can be executed using google colab or jupyter notebook

In this there's a file titled "Resume.csv". It has 4 columns. Resume ID, Resume_str, resume_html, and category. For the purpose of this assignment, I used ID and Resume_str data.
I used pandas library to clean the data. For tokenization and segmenting, I used Named Entity Recognition (NER) to extract sections like skills, experience, education, etc. NER will help identify different sections of the resume such as names, organizations, locations, and job titles. I used spaCy for this. This approach extracts important entities (e.g., names, companies, etc.) from each resume and can be used to segment resumes into meaningful sections.

Libraries and Frameworks used:
- FastAPI: Used to create the REST API. FastAPI provides a quick, high-performance web framework for building and testing endpoints.
- ChromaDB: A vector database that allows efficient storage and retrieval of embeddings. This tool is used to manage resume embeddings and perform fast similarity searches.
- SentenceTransformer: A pre-trained model library for creating embeddings from textual data. It converts resumes and user queries into vector representations for similarity matching.
- Transformers: The Hugging Face transformers library provides a pre-trained GPT model for generating responses based on the retrieved context.
- spaCy: A powerful Natural Language Processing (NLP) library used for Named Entity Recognition (NER) to extract important entities like skills, names, and companies from resumes.
- CrossEncoder: A specialized encoder for ranking and scoring retrieved documents in context with the user query, making the retrieval more precise.
- Uvicorn: A lightweight ASGI server for running the FastAPI application.

Embedding Generation
The second step involved converting cleaned resume text into vector embeddings:

Using SentenceTransformer models (all-MiniLM-L6-v2) to generate embeddings for the resumes.
- Embeddings capture the semantic information from the text, making it easier to perform similarity searches.

Challenges:
- Handling large embeddings: Managing and saving large embedding vectors required storing them efficiently using numpy.

To enable efficient retrieval of resume embeddings:
- ChromaDB was set up to store and manage embeddings in a vector database.
- Metadata (e.g., resume ID and category) was stored alongside the embeddings to allow for easy retrieval.

Challenges:
- Embedding formatting: Ensuring the embeddings were stored and queried in the right format was critical, especially when working with numpy arrays.
- Database indexing: I experimented with different indexing techniques (e.g., IVF) to improve retrieval speed, which was vital when handling large datasets.

**Retrieval-Augmented Generation (RAG) Pipeline**

The core of the project was building a pipeline that:

- Converts user queries into embeddings using SentenceTransformer.
- Queries ChromaDB to retrieve the most relevant resumes based on similarity.
- Ranks the retrieved results using a CrossEncoder model for better accuracy.
- Passes the top documents to a GPT-2 model for generating responses that answer the user’s query.
  
Challenges:
- Matching user queries to resumes: One challenge was creating accurate embeddings for user queries to match relevant resumes. The CrossEncoder improved precision in scoring relevant documents.
- Generating relevant responses: Making the GPT-2 model provide context-aware responses was a challenge, as language models sometimes produce generic or non-specific outputs.

**API Development with FastAPI**
To make the RAG pipeline accessible, I developed a REST API using FastAPI:

- The /query endpoint accepts a user query as input, passes it through the RAG pipeline, and returns a generated response.
- CORS Middleware was added to handle cross-origin requests.
- Handling API errors: One common issue was handling incorrect or missing input data, which resulted in 422 Unprocessable Entity errors. Using Pydantic models for input validation solved this.
- Exposing the API for external testing: Initially, I used ngrok to expose the FastAPI server, but due to account limitations, testing was challenging.

**Additional notes**
- Embedding and Retrieval Precision: Ensuring the embeddings accurately captured the semantic meaning of resumes and queries was crucial. The use of CrossEncoder to rank retrieved documents helped improve retrieval precision.

- GPT-2 Model Output Quality: While GPT-2 worked for basic generation, it sometimes struggled with very specific or complex queries. This could be improved by fine-tuning the model or using a larger model like GPT-3.
- Future improvements could include enhancing the retrieval accuracy further and deploying the API on a more stable environment like AWS or GCP for better scalability.





