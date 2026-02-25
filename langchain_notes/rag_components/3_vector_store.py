"""
* Why do we need Vector Stores?
Traditional relational databases (SQL) are excellent for keyword matching but fail at semantic search (understanding meaning).
    > Keyword Flaw: 
        A search for "Spider-Man" based on keywords like "Director: Sam Raimi" might recommend a romantic drama by the same director, which is a poor recommendation for an action fan.
    > Semantic Solution: 
        By comparing the plot (story) of movies using embeddings, the system can recommend films with similar themes even if they have different actors or directors (e.g., Taare Zameen Par and A Beautiful Mind share themes of mental brilliance and struggle).

* What is a Vector Store?
A Vector Store is a system designed to store and retrieve data represented as numerical vectors (embeddings).
    > Embeddings: 
        A technique that uses neural networks to convert text meaning into a list of numbers (vectors).
    > Storage Options:
        In-Memory: Stored in RAM (fast but temporary).
        On-Disk: Stored on a hard drive or database (persistent).

* Key Features of Vector Stores
    1. Storage: Stores the vector itself and associated Metadata (e.g., Movie ID, Name).
    2. Similarity Search: Compares a "query vector" against stored vectors to find the most similar results using measures like Cosine Similarity.
    3. Indexing: A smart method to make searching faster. Instead of comparing a query to 1 million vectors one by one (Linear Search), it uses clusters or "Approximate Nearest Neighbor" (ANN) lookups to narrow down the search space quickly.
    4. CRUD Operations: Supports Creating, Reading, Updating, and Deleting vectors.

* Vector Store vs. Vector Database
    > Vector Store: 
        A lightweight library (like FAISS) focused strictly on storage and similarity search. Ideal for prototypes.
    > Vector Database: 
        A full-fledged database system (like Pinecone, Milvus, ChromaDB) with enterprise features like distributed architecture, backups, security/authentication, and multi-user concurrency.

* Implementing ChromaDB in LangChain
Key Operations in Code:
    > Initialization: Requires an embedding model (e.g., OpenAIEmbeddings) and a persistence directory.
    > Adding Documents: Use add_documents(). LangChain assigns unique IDs to each chunk.
    > Similarity Search:  similarity_search(query, k=2) returns the top 2 most similar documents.
    > similarity_search_with_score() returns results along with their distance score (lower is better)
    > Metadata Filtering: You can filter results based on metadata fields (e.g., "Find players where team = 'CSK'").
    > Update/Delete: Documents can be updated or removed using their unique IDs.

* Why LangChain is useful for Vector Stores
LangChain provides a common interface for all major vector stores. This means you can write your code once and easily switch from ChromaDB to Pinecone or FAISS by just changing one line of code; the functions like similarity_search remain the same.
"""
