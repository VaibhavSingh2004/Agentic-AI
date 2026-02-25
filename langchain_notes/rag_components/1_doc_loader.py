"""
* Introduction to RAG (Retrieval-Augmented Generation)
RAG is a technique used to overcome the limitations of LLMs, such as training data cut-offs (lack of current info) and lack of access to private/personal data.

* How it works: 
    It connects an LLM to an external knowledge base (PDFs, databases, etc.). When a query is made, the system retrieves relevant info from this base and uses it as context for the LLM.
* Key Components of RAG:
    1. Document Loaders (Loading data from sources)
    2. Text Splitters (Breaking data into chunks)
    3. Vector Databases (Storing data for retrieval)
    4. Retrievers (Fetching relevant chunks)

* What are Document Loaders?
Document Loaders are LangChain components that load data from various sources into a standardized format called a Document Object.

* Structure of a Document Object:
    > Page Content: The actual text data.
    > Metadata: Information about the data (source, page number, author, etc.)
    > Storage: All loaders in LangChain typically return a list of Document Objects 

* Key Document Loaders Covered
    A. Text Loader
        > Purpose: Loads simple .txt files.
        > Code Tip: Use langchain_community.document_loaders.TextLoader. You can specify encoding (e.g., utf-8) if the file contains special characters.

    B. PyPDF Loader
        > Purpose: Loads .pdf files.
        > Mechanism: It works on a page-by-page basis. If a PDF has 25 pages, it creates 25 Document Objects.
        > Limitation: It is best for text-based PDFs. For scanned images or complex tables, you should use specialized loaders like PDFPlumber or Amazon Textract.

    C. Directory Loader
        > Purpose: Loads multiple files from a folder simultaneously.
        > Features: Supports patterns (e.g., .pdf to load only PDFs).
            Requires you to specify which loader class to use for the files (e.g., PyPDFLoader)

    D. WebBase Loader
        > Purpose: Extracts text from web pages (URLs).
        > Mechanism: Uses Requests and BeautifulSoup internally.
        > Limitation: Best for static sites (blogs, news). For JavaScript-heavy sites, use SeleniumURLLoader.

    E. CSV Loader
        > Purpose: Loads .csv files.
        > Mechanism: Each row in the CSV becomes a separate Document Object.

* Load vs. Lazy Load
    > load() (Eager Loading): Loads everything into RAM at once. Use this for small datasets
    > lazy_load(): Returns a generator. It loads one document at a time, processes it, and moves to the next. Use this for massive datasets (100s of PDFs) to save memory and avoid lag.

* Custom Document Loaders
If a source isn't supported by LangChain, you can create a custom loader by inheriting from the BaseLoader class and defining your own load and lazy_load logic.
"""
