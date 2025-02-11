# import os
from typing import Any, Dict, List

from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from tqdm import tqdm  # Import tqdm
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

from FileIO import save_summarization_results, ReadFile




FILE_PATH = "f:\Miles to Go (Cyrus Miley).epub"
CONTEXT_LENGTH=11989
# CONTEXT_LENGTH=11000
# CONTEXT_LENGTH=32768
OPENAI_API_BASE="http://localhost:1234/v1"
OPENAI_API_KEY="dummy_value"
MODEL_NAME="t-pro-it-1.0@q4_k_m"
# MODEL_NAME="t-pro-it-1.0@q5_k_m"


# Prompt templates (Provided in the prompt)
# reduce_prompt_template = """Ниже приведен набор резюме:
# {docs}
# Возьмите их и составьте окончательное, консолидированное резюме.
# РЕЗЮМЕ:"""

final_prompt_template = """Ниже приведен набор резюме:
{docs}
Возьмите их и составьте окончательное, консолидированное резюме.
РЕЗЮМЕ:"""
final_prompt = PromptTemplate(template=final_prompt_template, input_variables=["docs"])

reduce_prompt_template = """Ниже приведен набор резюме:
{docs}
Объедините и консолидируйте их для составления промежуточного резюме, с кратким пересказом сюжета.
ПРОМЕЖУТОЧНОЕ РЕЗЮМЕ:"""
reduce_prompt = PromptTemplate(template=reduce_prompt_template, input_variables=["docs"])

# map_prompt_template = """Ниже приведен набор документов
# {docs}
# Основываясь на этом списке документов, напишите краткое резюме.
# КРАТКОЕ РЕЗЮМЕ:"""
# map_prompt_template = """Перескажи вкратце представленный отрывок:
# {docs}"""
map_prompt_template = """Переведи на русский и перескажи вкратце (не более 100 слов) без повторов представленный отрывок:
{docs}"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["docs"])


set_llm_cache(SQLiteCache(database_path=".langchain.db"))

def detect_chunk_size_by_token_count(llm: ChatOpenAI, text: str, context_length: int) -> int:
    """
    Gets the token count for a given text using the LLM's API.
    """

    chunk_size=20000
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
         separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        ""
        ]
    )
    text = text_splitter.split_text(text)[0]

    num_tokens = llm.get_num_tokens(text)
    
    auto_chunk_size:int = (int)(context_length * chunk_size / num_tokens) #- (context_length * chunk_size / num_tokens) * 0.05)
    print(f"auto_chunk_size: {auto_chunk_size}")
    return auto_chunk_size

def load_and_split_text(text: str, chunk_size: int , chunk_overlap: int = 100):
    """
    Loads text from a file and splits it into chunks.

    Args:
        file_path: Path to the text file.
        chunk_size: The desired chunk size.
        chunk_overlap: The overlap between chunks.

    Returns:
        A list of Document objects containing the text chunks.  Returns an empty list if an error occurs.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            ""
            ]
        )
        texts = text_splitter.split_text(text) #[:3]
        print(f"Text splitted on: {len(texts)} pices")

        # for chunk in texts:
        #     chunk_token_count = get_token_count(chunk, llm)

        #  Convert text chunks into Langchain Document objects
        docs = [Document(page_content=t) for t in texts]
        return docs
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return []  # Return an empty list in case of an error
    except Exception as e:
        print(f"Error reading or splitting text: {e}")
        return []


# Summarization using Map Reduce

# def summarize_with_map_reduce(
#     llm: ChatOpenAI, docs: List[Document], map_prompt: PromptTemplate, reduce_prompt: PromptTemplate
# ):
#     """
#     Summarizes a list of documents using the map-reduce chain.

#     Args:
#         llm: The initialized LlamaCpp model.
#         docs: A list of Document objects.
#         map_prompt: The prompt for the map step.
#         reduce_prompt: The prompt for the reduce step.

#     Returns:
#         The final summarized text.
#     """
#     try:
#         callbacks = [StdOutCallbackHandler()]
#         # Define LLM Chains
#         map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=False)
#         reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt, verbose=True, callbacks=callbacks)

#         # Define StuffDocumentsChain
#         combine_documents_chain = StuffDocumentsChain(
#             llm_chain=reduce_chain, document_variable_name="docs"
#         )

#         # Actually do the summarization
#         intermediate_summaries = []
#         #  Wrap the loop with tqdm for progress bar
#         for doc in tqdm(docs, desc="Mapping chunks"):
#             intermediate_summary = map_chain.run(doc.page_content)
#             intermediate_summaries.append(intermediate_summary)

#         # Print intermediate summaries and their sizes
#         print("\nIntermediate Summaries:")
#         for i, summary in enumerate(intermediate_summaries):
#             # token_count = count_tokens(summary, tokenizer)
#             print(f"  Chunk {i+1}: {summary[:100]}")  # Show first 100 chars

#         # Combine intermediate summaries
#         final_summary = combine_documents_chain.run(
#             [Document(page_content=s) for s in intermediate_summaries]
#         )
#         return final_summary, intermediate_summaries

#     except Exception as e:
#         print(f"Error during summarization: {e}")
#         return None


def split_text_with_token_limit(llm: ChatOpenAI, text: str, context_length: int, overlap: int = 100):
    """Splits text into chunks respecting the model's token limit."""
    chunks = []
    current_chunk = ""
    splitter = RecursiveCharacterTextSplitter(
        chunk_overlap=overlap,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            ""
        ]
    )
    for piece in splitter.split_text(text):
        if llm.get_num_tokens(current_chunk + piece) <= context_length * 0.99: #Safety margin
            current_chunk += piece
        else:
            chunks.append(Document(page_content=current_chunk))
            current_chunk = piece  # Start a new chunk
    chunks.append(Document(page_content=current_chunk))  # Add the last chunk
    return chunks

class LoggingHandler(BaseCallbackHandler):
    # def on_chat_model_start(
    #     self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    # ) -> None:
    #     print("Chat model started")

    # def on_llm_end(self, response: LLMResult, **kwargs) -> None:
    #     print(f"Chat model ended, response: {response}")

    # def on_chain_start(
    #     self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    # ) -> None:
    #     if serialized is not None:
    #         print(f"Chain {serialized.get('name')} started")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print(f"Chain ended, outputs: {outputs}")

def summarize_with_map_reduce(
    llm, docs: List[Document], map_prompt: PromptTemplate, reduce_prompt: PromptTemplate, context_length: int
):
    """Summarizes with map-reduce, handling context overflow."""
    try:
        callbacks = [StdOutCallbackHandler()]
        # Define LLM Chains
        # map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=True, callbacks=callbacks)
        map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=False, callbacks=[LoggingHandler()])
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt, verbose=True, callbacks=callbacks)
        final_chain = LLMChain(llm=llm, prompt=final_prompt, verbose=True, callbacks=callbacks)

        # Define StuffDocumentsChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        ) 
        # final_chain2 = StuffDocumentsChain(
        #     llm_chain=final_chain, document_variable_name="docs"
        # )

        intermediate_summaries = []
        for doc in tqdm(docs, desc="Mapping chunks"):
            intermediate_summary = map_chain.invoke(doc.page_content)
            intermediate_summaries.append(intermediate_summary['text'])

        # Print intermediate summaries and their sizes
        print("\nIntermediate Summaries:")
        for i, summary in enumerate(intermediate_summaries):
            # token_count = count_tokens(summary, tokenizer)
            print(f"  Chunk {i+1}: {summary[:100]}")  # Show first 100 chars


        final_summary = ""
        intermediate_summaries_tokens = llm.get_num_tokens("\n\n".join(intermediate_summaries))
        if intermediate_summaries_tokens > context_length * 0.99:
            print(f'\nIntermediate summaries tokens {intermediate_summaries_tokens} are larger than context size {context_length}. Reducing summaries.')

            current_summary_batch = []
            reduced_summaries = []
            for doc in tqdm(intermediate_summaries, desc="Reducing summaries"):
                if llm.get_num_tokens("\n\n".join([d.page_content for d in current_summary_batch] + [doc])) > context_length * 0.99: # Check before adding
                    # Reduce the current batch
                    reduced_summaries.append(Document(page_content=combine_documents_chain.run(current_summary_batch)))
                    current_summary_batch = [] #Start a new batch
                current_summary_batch.append(Document(page_content=doc))
            final_summary = final_chain.run(reduced_summaries)
            
        else:
            final_summary = final_chain.run(intermediate_summaries)

        return final_summary, intermediate_summaries

    except Exception as e:
        print(f"Error during summarization: {e}")
        return None, None # Return None for both to avoid unpacking errors


# Main Execution

if __name__ == "__main__":
    try:
        llm = ChatOpenAI(
            openai_api_base=OPENAI_API_BASE,
            openai_api_key=OPENAI_API_KEY,
            model_name=MODEL_NAME,
            temperature=0,
            # presence_penalty=2,
            # """Penalizes repeated tokens."""
            # frequency_penalty=2
            # repeat_penalty=2
            # """Penalizes repeated tokens according to frequency."""
            # max_tokens=4000,  # Limit tokens for each response
            # max_retries=2,  # Retry on failure
        )

        text = ReadFile(FILE_PATH)

        # auto_chunk_size = detect_chunk_size_by_token_count(llm, text, context_length=CONTEXT_LENGTH)
        # docs = load_and_split_text(text, chunk_size=auto_chunk_size ) #* 0.95)
        # docs = load_and_split_text(text, chunk_size=auto_chunk_size * 0.95)

        docs = split_text_with_token_limit(llm, text, CONTEXT_LENGTH)
          

        if docs:  # Only proceed if documents were loaded successfully
            
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(
                    content="Отвечай на русском."
                ),
                HumanMessage(
                    content="What is your name?"
                )
            ]
            # Define a chat model and invoke it with the messages
            print(llm.invoke(messages))

            # from langchain.chains.summarize import load_summarize_chain
            # chain = load_summarize_chain(llm, chain_type="map_reduce", token_max=CONTEXT_LENGTH)
            # summarize_chain = chain.run(docs)

            final_summary, intermediate_summaries = summarize_with_map_reduce(
                llm, docs, map_prompt, reduce_prompt, CONTEXT_LENGTH
            )

            if final_summary:
                print("Summary:")
                print(final_summary)

                  # Save the results
                save_summarization_results(FILE_PATH, final_summary, intermediate_summaries)
            else:
                print("Summarization failed.")

        else:
            print("No documents to summarize. Check the input file and loading process.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")