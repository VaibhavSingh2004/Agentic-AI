from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
? https://www.analyticsvidhya.com/blog/2024/07/langchain-text-splitters/

* What is Text Splitting?
Text splitting is the process of dividing large pieces of text (articles, PDFs, books) into smaller, manageable chunks that an LLM can process effectively.

* Why is it important?
    > Model Limitations: 
        Most LLMs have a context length limit (e.g., 50,000 tokens). Large files must be split to fit within these bounds.
    > Better Embedding Quality: 
        Small, focused chunks capture semantic meaning more precisely than large, mixed-topic documents.
    > Precise Semantic Search: 
        Searching through smaller chunks yields more relevant results compared to searching through one giant file.
    > Resource Optimization: 
        Smaller chunks are more memory-efficient and allow for parallel processing.

* Main Types of Text Splitters in LangChain
    A. Length-Based Splitting (Character Text Splitter)
        The simplest method where text is split strictly based on a character count.
        > Pros: Fast and easy to implement.
        > Cons: Often breaks text in the middle of words or sentences, losing linguistic structure.
        > Key Parameter - Chunk Overlap: To prevent context loss at the break points, you can overlap chunks (e.g., 10-20% overlap). This keeps some information from the previous chunk in the next one.

    B. Text Structure-Based (Recursive Character Text Splitter)
        This is the most widely used splitter. It tries to split text based on a hierarchy of characters: Paragraphs (\n\n) → Sentences (\n) → Words ( ) → Characters.
        > Mechanism: 
            It first attempts to split by paragraphs. If a paragraph is still larger than the chunk_size, it tries to split it into sentences, then words, until it fits the limit.
        > Goal: To keep related text together and avoid breaking words in half.

    C. Document-Based Splitting
        Used for non-plain text files like Code (Python, Java, etc.), Markdown, or HTML
        > Custom Separators: 
            Instead of just newlines, it uses language-specific keywords (e.g., class, def in Python) to split chunks logically according to the file structure.
        > LangChain Support: 
            You can create these by calling RecursiveCharacterTextSplitter.from_language and specifying the target language.

    D. Semantic Meaning-Based (Semantic Chunker)
        A newer, experimental approach that splits text based on the meaning rather than the structure.
        > Mechanism: 
            It creates embeddings for every sentence and measures the cosine similarity between consecutive sentences. When a sudden drop in similarity occurs (indicating a topic change), it triggers a split.
        > Status: 
            Currently experimental in LangChain and can be slower and less accurate than structural splitters, but highly promising for mixed-topic documents.


* Summary Table

| Splitter Type         | Best Use Case                 | Logic                                 |
| --------------------- | ----------------------------- | ------------------------------------  |
| Character             | Very simple, fast tasks       | Fixed character count                 |
| Recursive Character   | Standard RAG applications     | Hierarchy: Para > Sentence > Word     |
| Code/Markdown         | Programming or technical docs | Language-specific keywords            |
| Semantic              | Mixed-topic documents         | Meaning similarity between sentences  |
"""

text = """
    LangChain Text Splitters are essential for handling large documents by breaking them into manageable chunks. This improves performance, enhances contextual understanding, allows parallel processing, and facilitates better data management. Additionally, they enable customized processing and robust error handling, optimizing NLP tasks and making them more efficient and accurate. Further, we will discuss methods to split data into manageable chunks.
    LangChain Text Splitters are essential for handling large documents by breaking them into manageable chunks. This improves performance, enhances contextual understanding, allows parallel processing, and facilitates better data management. Additionally, they enable customized processing and robust error handling, optimizing NLP tasks and making them more efficient and accurate. Further, we will discuss methods to split data into manageable chunks.
"""


char_text_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=50,
    chunk_overlap=0,
    is_separator_regex=False,
)

# when load data using any loader class use this method to split text
# result = char_text_splitter.split_documents(text)

"""
This function splits the text where each chunk has a maximum of 500 characters. Text will be split only at new lines since we are using the new line (“\n”) as the separator. If any chunk has a size more than 500 but no new lines in it, it will be returned as such.
"""
result = char_text_splitter.split_text(text)
print(result[0])

from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
Rather than using a single separator, we use multiple separators. This method will use each separator sequentially to split the data until the chunk reaches less than chunk_size. We can use this to split the text by each sentence,
"""

#  third seperator is for splitting by sentence using regex.
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", r"(?<=[.?!])\s+", " "],
    keep_separator=False,
    is_separator_regex=True,
    chunk_size=100,
    chunk_overlap=0,
)

result = recursive_splitter.split_text(text)
len(result)


# a few sample chunks
for t in result:
    print(len(t))
    print(t)

# ----------------------------------------------
# JSON Recursive Text Splitter
# ----------------------------------------------

from langchain_text_splitters import RecursiveJsonSplitter

# Example JSON object
json_data = {
    "company": {
        "name": "TechCorp",
        "location": {"city": "Metropolis", "state": "NY"},
        "departments": [
            {
                "name": "Research",
                "employees": [
                    {"name": "Alice", "age": 30, "role": "Scientist"},
                    {"name": "Bob", "age": 25, "role": "Technician"},
                ],
            },
            {
                "name": "Development",
                "employees": [
                    {"name": "Charlie", "age": 35, "role": "Engineer"},
                    {"name": "David", "age": 28, "role": "Developer"},
                ],
            },
        ],
    },
    "financials": {"year": 2023, "revenue": 1000000, "expenses": 750000},
}

"""
-----------------------------------------------------------------------------------------------------
A nested json object can be split such that initial json keys are in all the related chunks of text. 
If there are any long lists inside, we can convert them into dictionaries to split.
-----------------------------------------------------------------------------------------------------
This splitter maintains initial keys such as company and departments 
if the chunk contains data corresponding to those keys.
-----------------------------------------------------------------------------------------------------
"""

splitter = RecursiveJsonSplitter(max_chunk_size=200, min_chunk_size=20)
chunks = splitter.split_text(json_data, convert_lists=True)
# chunks = splitter.split_json(json_data, convert_lists=True)

"""
Sample Output: when we call split_json
-----------------------------------------------------------------------------------------------------
[{'company': {'name': 'TechCorp',
   'location': {'city': 'Metropolis', 'state': 'NY'}}},
 {'company': {'departments': {'0': {'name': 'Research',
     'employees': {'0': {'name': 'Alice', 'age': 30, 'role': 'Scientist'},
      '1': {'name': 'Bob', 'age': 25, 'role': 'Technician'}}}}}},
 {'company': {'departments': {'1': {'name': 'Development',
     'employees': {'0': {'name': 'Charlie', 'age': 35, 'role': 'Engineer'},
      '1': {'name': 'David', 'age': 28, 'role': 'Developer'}}}}}},
 {'financials': {'year': 2023, 'revenue': 1000000, 'expenses': 750000}}]

 
Sample Output: When we call split_text
-----------------------------------------------------------------------------------------------------
Length: 84
{"company": {"name": "TechCorp", "location": {"city": "Metropolis", "state": "NY"}}}
Length: 183
{"company": {"departments": {"0": {"name": "Research", "employees": {"0": {"name": "Alice", "age": 30, "role": "Scientist"}, "1": {"name": "Bob", "age": 25, "role": "Technician"}}}}}}
Length: 188
{"company": {"departments": {"1": {"name": "Development", "employees": {"0": {"name": "Charlie", "age": 35, "role": "Engineer"}, "1": {"name": "David", "age": 28, "role": "Developer"}}}}}}
Length: 70
{"financials": {"year": 2023, "revenue": 1000000, "expenses": 750000}}
-----------------------------------------------------------------------------------------------------
"""
# Process the chunks as needed
for chunk in chunks:
    print(len(chunk))
    print(chunk)
