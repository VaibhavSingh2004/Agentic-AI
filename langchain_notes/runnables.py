"""
* Evolution of LangChain: Why we need Runnables ?
    > Early Days (2022): LangChain started by creating simple wrappers (classes) to talk to different LLM providers (OpenAI, Anthropic, Google) with minimal code changes.
    > Component Explosion: They realized building LLM apps involves more than just API calls. They added components for:
        > Document Loaders: To bring in data.
        > Text Splitters: To break down long documents.
        > Vector Stores & Retrievers: For semantic search.
        > Output Parsers: To clean LLM responses.

? The Rise of "Chains": To simplify manual connections between these components, LangChain introduced built-in functions like LLMChain and RetrievalQA.
? The Problem: Over time, they created too many specific chains, leading to a heavy code base and a steep learning curve for developers. The components weren't standardized; some used .predict(), others .format() or .get_relevant_documents().

* What are Runnables?
A unit of work that can be invoked, batched, streamed, transformed and composed.
Runnables were introduced to standardize how components interact.
Standard Interface: Every Runnable follows the same interface with core methods:
    > invoke(): Process a single input.
    > batch(): Process multiple inputs efficiently.
    > stream(): Stream the output as it's generated.

Lego Block Analogy: Runnables are like Lego blocks. Because they share the same "connectors" (the standard interface), you can snap them together in any order to build complex workflows.

* Building a Runnable from Scratch (Conceptual)
Abstract Base Class: A Runnable class is created using Python's ABC module. It defines a mandatory invoke method.

Standardizing Components:
    > LLM Class: Instead of just .predict(), it now implements .invoke().
    > Prompt Template: Instead of just .format(), it now implements .invoke().
    > The Connector: A RunnableConnector class is built to loop through a list of Runnables, passing the output of the first as the input to the next.

* Power of Composition
With the Runnable standard, you can:
    > Chain Components: Connect a Prompt → LLM → Output Parser seamlessly.
    > Chain Chains: You can take one chain (e.g., a Joke Generator) and connect it to another chain (e.g., a Joke Explainer) to create a larger, unified workflow.

* Types of Runnables
Runnables are divided into two main categories:
    ?> Task-Specific Runnables:
        These are core LangChain components converted into Runnables (e.g., ChatOpenAI, PromptTemplate, OutputParser). They have a specific purpose, like talking to an LLM or designing a prompt.
    ?> Runnable Primitives:
        These are fundamental building blocks that define the execution logic. They help orchestrate how different Task-Specific Runnables interact (sequentially, parallelly, or conditionally).

* Core Runnable Primitives
    A. RunnableSequence
        Purpose: Connects multiple Runnables in a sequence where the output of one becomes the input of the next.
        Implementation: It is the backbone of most LangChain apps.
        Example: Prompt → Model → Parser.

    B. RunnableParallel
        Purpose: Executes multiple Runnables concurrently on the same input.
        Key Behavior: Each branch receives the same input independently and produces a dictionary of outputs.
        Example: Generating a Tweet and a LinkedIn post about the same topic simultaneously.

    C. RunnablePassthrough
        Purpose: A special primitive that passes the input to the output unchanged without any processing.
        Use Case: Useful when you want to keep the original data (like a generated joke) while also processing it in a parallel branch (like generating an explanation).

    D. RunnableLambda
        Purpose: Converts any standard Python function into a Runnable.
        Advantage: Allows you to inject custom logic (like text cleaning, word counting, or pre-processing) directly into a LangChain pipeline.
        Example: Creating a custom "word counter" function and including it in a chain.

    E. RunnableBranch
        Purpose: Used to create conditional workflows (the "If-Else" of LangChain).
        Implementation: It takes a list of (condition, runnable) pairs and a final default runnable.
        Example: If a report is >300 words, trigger a "Summary" chain; otherwise, pass the original report as is.

* LangChain Expression Language (LCEL)
    The Pipe Operator (|): This is the core syntax. It overloads the Python __or__ method to create a RunnableSequence.
    Syntax: chain = prompt | model | parser
    Unified Interface: Every LCEL chain automatically inherits sync, async, batch, and streaming support without extra code.
"""

# imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------
# RunnableSequence
# --------------------------------------
prompt1 = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
parser = StrOutputParser()
prompt2 = PromptTemplate(
    template="Explain the following joke - {text}", input_variables=["text"]
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
# another syntax of RunnableSequence
# chain = prompt1 | model | parser | prompt2 | model | parser
print(chain.invoke({"topic": "AI"}))


# --------------------------------------
# RunnableParllel
# --------------------------------------
prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}", input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Generate a Linkedin post about {topic}", input_variables=["topic"]
)

parser = StrOutputParser()
parallel_chain = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt1, model, parser),
        "linkedin": RunnableSequence(prompt2, model, parser),
    }
)

result = parallel_chain.invoke({"topic": "AI"})

print(f"Tweet: {result['tweet']}")
print(f"------" * 10)
print(f"LinkedIn Post: {result['linkedin']}")


# --------------------------------------
# RunnableBranch and RunnablePassThrough
# --------------------------------------
prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}", input_variables=["text"]
)


parser = StrOutputParser()
report_gen_chain = prompt1 | model | parser
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | model | parser),
    RunnablePassthrough(),
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({"topic": "Russia vs Ukraine"}))
# for chunk in final_chain.stream({"topic": "Russia vs Ukraine"}):
#     print(chunk, end="", flush=True)


# --------------------------------------
# RunnableLambda
# --------------------------------------
def word_count(text):
    return len(text.split())


prompt = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)


parser = StrOutputParser()
joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel(
    {"joke": RunnablePassthrough(), "word_count": RunnableLambda(word_count)}
)
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({"topic": "AI"})

final_result = """{} \n word count - {}""".format(
    result["joke"], result["word_count"]
)

print(final_result)

# for seeeing the visual diagram of chains
# print(final_chain.get_graph().print_ascii())
# Example Output:
"""
              +-------------+
              | PromptInput |
              +-------------+
                      *
                      *
                      *
             +----------------+
             | PromptTemplate |
             +----------------+
                      *
                      *
                      *
         +------------------------+
         | ChatGoogleGenerativeAI |
         +------------------------+
                      *
                      *
                      *
            +-----------------+
            | StrOutputParser |
            +-----------------+
                      *
                      *
                      *
     +--------------------------------+
     | Parallel<joke,word_count>Input |
     +--------------------------------+
              **            ***
            **                 **
          **                     **
+-------------+              +------------+
| Passthrough |              | word_count |
+-------------+              +------------+
              **            ***
                **        **
                  **    **
    +---------------------------------+
    | Parallel<joke,word_count>Output |
    +---------------------------------+
"""
