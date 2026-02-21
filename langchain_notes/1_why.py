# Why we need Langchain ?

"""
? What is LangChain?
    -> LangChain is an open-source framework designed for developing applications that are powered by LLMs
       (Large Language Models).
    -> In essence, an LLM handles the "heavy lifting" (understanding language and generating text), but  
       LangChain is the tool that orchestrates all the necessary steps and components required to build a complete, end-to-end application around the LLM.

? Why LangChain is Needed: The Problem of Orchestration
The need for LangChain is best understood by looking at the challenges in building an application like a Q&A system for large documents (like a 1000-page book) using an LLM:

! The Challenges Solved by LLMs and APIs
    * The "Brain" Challenge (NLU and Text Generation):
    -> Developing a component to understand a user's question (Natural Language Understanding) and generate 
       a relevant, context-aware answer from a document was historically very challenging.
    -> Solution: LLMs (like those from the GPT family) have solved this problem by providing the necessary 
       NLU and text generation capabilities.

    * The Computation and Cost Challenge:
    -> Hosting a massive LLM model directly on your own server for inference is computationally intensive and 
       very expensive.
    -> Solution: LLM providers created APIs around their models, allowing developers to use the LLM's power 
       without hosting it, reducing computational burden and enabling pay-per-use cost models.

! The Core Challenge Solved by LangChain: Orchestration
Even with LLMs and their APIs, building a complete, functioning system requires a complex pipeline involving many different technologies:

    -> Components/Moving Parts: Document Storage (like AWS S3), Document Loader, Text Splitter, Embedding
       Model (to convert text to vectors), Vector Database (to store vectors for semantic search), and the LLM API.

    -> Tasks: Loading documents, splitting them into small chunks, creating embeddings, storing them,  
       retrieving the most relevant chunks based on a query (Semantic Search), and finally sending the query and context to the LLM .

! The Key Problem: 
    -> Manually coding the interaction and data flow between all these disparate components is a 
       complex, error-prone, and time-consuming process. Furthermore, changing any component (e.g., swapping one vector database for another, or one LLM for a competitor's) requires rewriting significant portions of the code.

    -> LangChain's Solution: LangChain provides a framework that handles this orchestration, allowing you to 
       "plug and play" different components and manage the sequence of tasks (the pipeline) easily.
"""
