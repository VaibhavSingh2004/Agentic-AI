from dotenv import load_dotenv

load_dotenv()

# Components of Langchain

"""
1) Models
2) Prompts
3) Chains
4) Indexes
5) Memory
6) Agents
"""

# Models
# Documentation: https://docs.langchain.com/oss/python/langchain/models
"""
In Langchain, "models" are the core interfaces through which you interact with AI models.

! Two types of models we can use in langchain:
? language models (input: text, output: text)
   language models are further classified into two types of models:
   1) LLMs (text generator, Training data: General text corpora, No build-in memory, No undertanding of user and assistant roles)
   2) Chat Models (specialized for conversational tasks, Trainig Data: fine-tuned on chat datasets, Supports structured conversation history, Understands System, user and assistant roles)
? embeddings models (input: text, output: vectors)
"""
# --------------------------
# Code for language model
# --------------------------
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
result = llm.invoke("What is the capital of India")
print(result)

# --------------------------
# Code for OpenAI Chat model
# --------------------------
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4", temperature=1.5, max_completion_tokens=10)
result = model.invoke("Write a 5 line poem on cricket")
print(result.content)

# --------------------------
# Code for Anthropic Chat model
# --------------------------
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
result = model.invoke("What is the capital of India")
print(result.content)

# --------------------------
# Code for Google Chat model
# --------------------------
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
result = model.invoke("What is the capital of India")
print(result.content)

# --------------------------
# Code for HuggingFace API Chat model
# --------------------------
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India")
print(result.content)

# --------------------------
# Code for Huggingface Local Chat model
# --------------------------
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# this will set directory where models and all other files will be downloaded
os.environ["HF_HOME"] = "D:/huggingface_cache"

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.5, max_new_tokens=100),
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India")
print(result.content)

# --------------------------
# Code for Embeddings Models (OpenAI)
# --------------------------
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
# For generating embedding of single query use embed_query method
result = embedding.embed_query("Delhi is the capital of India")
print(str(result))

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France",
]
# For generating embedding of list of query use embed_documents method
result = embedding.embed_documents(documents)
print(str(result))

# --------------------------
# Code for Embeddings Models (Using Huggingface on Local System)
# --------------------------
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France",
]
vector = embedding.embed_documents(documents)
print(str(vector))

# Prompts
"""
Prompts are the input instruction or queries given to a model to guide its output.
A prompt is the message or instruction sent to a Large Language Model (LLM).
    Types: Text based prompts and Multi-Modal Prompts
    LLM outputs are highly sensitive to prompt wording. This has led to the emergence of "Prompt Engineering" as a specialized field.

! Static vs. Dynamic Prompts
? Static Prompts: 
Hard-coded messages sent directly by the programmer. These are risky in production because users might provide inconsistent or incorrect inputs, leading to undesirable LLM "hallucinations".

? Dynamic Prompts: 
These use templates with placeholders. Instead of asking the user for a full prompt, the application asks for specific variables (e.g., paper name, style) and injects them into a pre-defined template.

! PromptTemplate Class: 
LangChain provides the PromptTemplate class to manage these dynamic inputs.
? Why We need these prompts class ? Why we can not directly use f-string ?
Validation: It ensures all required placeholders are filled before sending the request.
Reusability: Templates can be saved as JSON files and reused across different parts of an application.
Integration: They fit seamlessly into LangChain "Chains," allowing multiple steps to be executed with a single invoke() call.

! Building a Chatbot & Managing Context
? The Context Problem: 
Basic LLM calls are stateless; they don't remember previous messages in a conversation.
? Chat History: 
To solve this, developers must maintain a list of past messages and send the entire history back to the LLM with every new query.
? LangChain Message Types: 
To help the LLM distinguish who said what, LangChain uses three message classes:
    SystemMessage: Sets the behavior/persona of the AI (e.g., "You are a helpful doctor").
    HumanMessage: Represents the user's input.
    AIMessage: Represents the response generated by the LLM.

! Advanced Prompt Templates
? ChatPromptTemplate: 
Similar to PromptTemplate, but designed for list-based chat conversations. It allows you to create dynamic System and Human messages within a single template.
? MessagePlaceholder: 
A special tool used within a ChatPromptTemplate to dynamically insert a variable-length chat history (often retrieved from a database) into the prompt at runtime. This is crucial for maintaining long-term context in customer support bots.
"""
# Example of propmt template
from langchain_core.prompts import PromptTemplate

# template
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  
1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True,
)

template.save("template.json")


# Example of Different types of messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about LangChain"),
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)

# Example of chat template
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful {domain} expert"),
        ("human", "Explain in simple terms, what is {topic}"),
    ]
)

prompt = chat_template.invoke({"domain": "cricket", "topic": "Dusra"})
print(prompt)

# Example of MessagePlaceholders
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template
chat_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful customer support agent"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
    ]
)

chat_history = []
# load chat history
with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())

print(chat_history)

# create prompt
prompt = chat_template.invoke(
    {"chat_history": chat_history, "query": "Where is my refund"}
)
print(prompt)

# Indexes
"""
Index Connect your application to external knowledge -- such as PDFs, websites or databases.
4 Main Components of Indexes:
1) doc loader
2) text splitter
3) vector store
4) retrivers
"""

# Memory
"""
LLM API Calls are stateless.
1st Question: Who is Virat Kohli?
2nd Question: How old he is? (LLM will not be able to answer this question because LLM did not remember the first question during API Calls.)

Different Types of Memory:
1) Conversation Buffer Memory: Stores a transcript of recent messages. Great for short chats but can grow large quickly.

2) Conversation Buffer Window Memory: Only keeps the last N interactions to avoid excessive token usage.

3) Summarizer-Based Memory: Periodically summarizes older chat segements to keep a condensed memory footprint.

4) Custom Memory: For advanced use cases, you can store specialized state (e.g., the user's preferences or key facts about them) in a custom memory class.
"""
