from dotenv import load_dotenv

load_dotenv()
#! -------------------------------------
#! Structured Output in Langchain
#! -------------------------------------

"""
! Introduction to Structured Output
? Concept:
Standard LLM outputs are unstructured (plain text).
Structured output involves forcing the model to return responses in a well-defined format, like JSON.
* Why it's important: Text output is great for humans, but machines and databases need structure.
* Structured output allows LLMs to talk to:
    > Databases: Storing extracted information.
    > APIs: Providing data to other software systems.
    > Agents: Passing specific parameters to tools (like a calculator).

? Key Use Cases
    > Data Extraction: Extracting specific fields (Name, University, CGPA) from resumes or documents into a database.
    > Review Analysis: Taking long product reviews and extracting specific keys like Summary, Pros, Cons, and Sentiment.
    > Building AI Agents: For an agent to use a tool, it must extract data in a precise format (e.g., extracting "2" from "find the square root of 2" to pass to a math tool).

! Methods to Define Output Schemas
LangChain provides a function called `with_structured_output()` to define the desired data format.
There are three main ways to specify this schema:

* A. TypedDict (Python Native)
    > What it is: A way to define a dictionary structure in Python where you specify keys and their data types.
    > Pros: Simple, uses standard Python syntax.
    > Cons: No data validation (if you define a field as an integer but the model returns a string, Python won't stop it).
    > Best for: Internal Python projects where strict validation isn't the priority.

* B. Pydantic (Recommended)
    > What it is: A data validation and parsing library for Python.
    > Key Features:
        > Validation: Throws an error if the data type doesn't match.
        > Field Constraints: You can set limits (e.g., CGPA must be between 0 and 10).
        > Descriptions: You can add descriptions to fields to help the LLM understand what to extract.
        > Best for: Most production-level Python applications.

* C. JSON Schema
    > What it is: A universal, language-agnostic data format.
    > Pros: Works across different programming languages (e.g., Python backend with a JavaScript frontend).
    > Best for: Projects requiring cross-language compatibility.

! Technical Implementation (The Workflow)
    > Define Schema: Create a class (Pydantic/TypedDict) or a JSON object defining your keys (e.g., summary, sentiment).
    > Initialize Model: Create your LLM instance (e.g., ChatOpenAI).
    > Bind Schema: Use model.with_structured_output(YourSchemaClass).
    > Invoke: Call the model. Instead of a text string, it returns an object/dictionary with your defined keys.

! Important Distinctions
    > Model Compatibility: Not all models support structured output by default.
    > Support: OpenAI (GPT-4o), Anthropic (Claude), and Google (Gemini) generally support it via "function calling" or "JSON mode."
    > No Support: Older or smaller open-source models (like TinyLlama) may fail and require Output Parsers.
    ? Methods:
        > Function Calling: Best for OpenAI models (default).
        > JSON Mode: Often used for models like Gemini or Claude.
"""

REVIEW = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
"""
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


# TypeDict schema
class ReviewTD(TypedDict):

    key_themes: Annotated[
        list[str],
        "Write down all the key themes discussed in the review in a list",
    ]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[
        Literal["pos", "neg"],
        "Return sentiment of the review either negative, positive or neutral",
    ]
    pros: Annotated[
        Optional[list[str]], "Write down all the pros inside a list"
    ]
    cons: Annotated[
        Optional[list[str]], "Write down all the cons inside a list"
    ]
    reviewer_name: Annotated[Optional[str], "Write the name of the reviewer"]


# Pydantic Schema
class ReviewPydantic(BaseModel):

    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(
        description="Return sentiment of the review either negative, positive or neutral"
    )
    pros: Optional[list[str]] = Field(
        default=None, description="Write down all the pros inside a list"
    )
    cons: Optional[list[str]] = Field(
        default=None, description="Write down all the cons inside a list"
    )
    name: Optional[str] = Field(
        default=None, description="Write the name of the reviewer"
    )


"""
* cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student')
This is How we add validations in Pydantic

* student_json = student.model_dump_json()
This is how we conver pydantic object into json (python-dict)
"""

# Json Schema
ReviewJS = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Write down all the key themes discussed in the review in a list",
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review",
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "Return sentiment of the review either negative, positive or neutral",
        },
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all the pros inside a list",
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all the cons inside a list",
        },
        "name": {
            "type": ["string", "null"],
            "description": "Write the name of the reviewer",
        },
    },
    "required": ["key_themes", "summary", "sentiment"],
}

structured_model = model.with_structured_output(ReviewJS)

result = structured_model.invoke(REVIEW)

print(type(result))


#! -------------------------------------
#! Output Parser in Langchain
#! -------------------------------------

"""
! Introduction to Output Parsers
Concept: LLMs return unstructured text (raw string + metadata). 
Output Parsers are classes in LangChain that convert this raw response into structured formats like JSON, CSV, Pydantic objects, or plain strings.
? The Problem: Unstructured text cannot be directly fed into databases or APIs. Output Parsers bridge this gap.

* Relationship to Previous Topic: While some models (OpenAI, Gemini) have built-in with_structured_output, many open-source models (like those on Hugging Face) do not. Output Parsers are essential for getting structured data from models that don't support it natively.

! Four Essential Output Parsers
* A. String Output Parser [06:32]
    > Function: Extracts only the text content from the LLM's complex response object, discarding metadata like token usage.
    > Use Case: Ideal for Chains. It allows you to pass the text output of one LLM directly as the input to the next step in a pipeline (chain).
    > Example Pipeline: Topic → Report Generation → String Output Parser → Summary Generation.

* B. JSON Output Parser [21:41]
    > Function: Instructs the LLM to return a JSON object and parses it into a Python dictionary.
    > Limitation: It does not enforce a specific schema. The LLM decides the keys and structure, which can lead to inconsistency.
    > Implementation: Requires calling get_format_instructions() and passing them into the prompt so the LLM knows to return JSON.

* C. Structured Output Parser [32:50]
    > Function: Extends the JSON parser by allowing you to define a schema (specific keys and descriptions).
    > Benefit: Enforces structure. You can mandate that the LLM provides specific keys (e.g., fact_1, fact_2).
    > Drawback: It does not perform data validation. If you ask for an integer and the LLM returns a string, it won't throw an error or correct it.

* D. Pydantic Output Parser [42:04]
    > Function: The most powerful parser. It uses Pydantic models to define the schema.
    > Features:
        > Strict Enforcement: Ensures the output matches the defined structure.
        > Data Validation: Checks data types (e.g., ensuring "Age" is an integer) and constraints (e.g., "Age" must be > 18).
        > Type Coercion: Can automatically convert a string like "35" into an integer 35.

! Implementation Workflow
    > Regardless of the parser, the workflow generally follows these steps:
    > Define Parser: Initialize the parser (and the schema if using Structured/Pydantic).
    > Get Format Instructions: Use parser.get_format_instructions() to get the text the LLM needs to see.   
    > Prompt Template: Create a template that includes a variable for these instructions.
    > Chain/Pipeline: Combine the Template → Model → Parser.
    > Invoke: Call the chain. The output will be the final parsed object (string or dictionary).

"""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

hf_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.7,
    provider="hf-inference",
)

hf_model = ChatHuggingFace(llm=hf_llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=["text"],
)

# This is the flow without StrOutput Format
prompt1 = template1.invoke({"topic": "black hole"})
result = model.invoke(prompt1)
prompt2 = template2.invoke({"text": result.content})
result1 = model.invoke(prompt2)
print(result1.content)

# With StrOutput parser we can do like this
parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "black hole"})
print(result)

# -------------------------------------------
# StructuredOutputParse
# -------------------------------------------

# StructuredOutputParser is Removed in latest version
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# schema = [
#     ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
#     ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
#     ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
# ]

# parser = StructuredOutputParser.from_response_schemas(schema)

# template = PromptTemplate(
#     template="Give 3 fact about {topic} \n {format_instruction}",
#     input_variables=["topic"],
#     partial_variables={"format_instruction": parser.get_format_instructions()},
# )

# chain = template | model | parser
# result = chain.invoke({"topic": "black hole"})
# print(result)

# -------------------------------------------
# PydanticOutputParse
# -------------------------------------------
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Person(BaseModel):

    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="Name of the city the person belongs to")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = template | hf_model | parser
final_result = chain.invoke({"place": "sri lankan"})
print(final_result)

# -------------------------------------------
# JsonOutputParse
# -------------------------------------------
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me 5 facts about {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = template | model | parser
result = chain.invoke({"topic": "black hole"})
print(result)


# from huggingface_hub import InferenceClient

# client = InferenceClient(
#     model="google/flan-t5-small",  # very small open model
#     token="hf_access_token",
# )

# print(client.text_generation("Hello"))
