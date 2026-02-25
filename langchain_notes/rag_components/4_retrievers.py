"""
* What is a Retriever?
A retriever is a component in LangChain that fetches relevant documents from a data source in response to a user's query.
    Input: User query.
    Output: Multiple LangChain Document objects.
    Nature: Unlike simple search, retrievers are Runnables, meaning they can be easily plugged into LangChain Expression Language (LCEL) chains.

* Categories of Retrievers
Retrievers can be categorized based on two main factors:
    ? Data Source: Where the data is coming from (e.g., Wikipedia, Vector Stores, Arxiv).
    ? Search Strategy: The mechanism used to find documents (e.g., Semantic search, MMR, Multi-query).

* Key Types of Retrievers
    A. Wikipedia Retriever
        > Function: Queries the Wikipedia API to fetch relevant content.
        > Mechanism: Uses keyword-based matching (not semantic search) to find the most relevant articles.
        > Use Case: Good for factual information lookup without needing a pre-built local database.

    B. Vector Store Retriever
        > Function: The most common retriever type; it fetches documents from vector databases like Chroma or FAISS.
        > Mechanism: Uses vector embeddings and semantic similarity to find content that is conceptually related to the query.
        > Benefit: Allows the system to understand the context of the query rather than just matching keywords.

    C. Maximum Marginal Relevance (MMR) Retriever
        > Problem Solved: Prevents "redundancy." Standard searches often return documents that say the exact same thing.
        > Mechanism: It balances relevance to the query with diversity among the results. It picks a relevant document, then picks the next one that is relevant but also very different from the first.
        > Parameter: Uses a lambda (0 to 1). 1 acts like standard similarity search; 0 provides maximum diversity.

    D. Multi-Query Retriever
        > Problem Solved: Handles ambiguous queries. If a user asks a vague question like "How to stay healthy?", a single search might miss relevant data.
        > Mechanism: It uses an LLM to generate multiple versions (different perspectives) of the user's query. It runs all these queries against the vector store and merges the unique results.

    E. Contextual Compression Retriever
        > Problem Solved: Avoids passing irrelevant "filler" text to the LLM. Often, a retrieved document is a large paragraph, but only one sentence is actually relevant.
        > Mechanism: It first retrieves documents and then uses an LLM to compress/trim them, keeping only the specific lines relevant to the user's question.

* Why Use Multiple Retrievers?
    Performance:
        Basic RAG systems often perform poorly. Switching to "Advanced RAG" by using specialized retrievers (like MMR or Contextual Compression) is the primary way to improve the accuracy and efficiency of GenAI applications.
    Cost & Context Window:
        Compressing documents saves money and prevents hitting LLM token limits.
"""
