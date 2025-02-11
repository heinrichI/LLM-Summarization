import os
from typing import List

from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from tqdm import tqdm  # Import tqdm
from langchain_core.callbacks.stdout import StdOutCallbackHandler
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from FileIO import save_summarization_results, ReadFile
from langchain.chains import (
                StuffDocumentsChain,
                LLMChain,
                ReduceDocumentsChain,
                MapReduceDocumentsChain,
            )

FILE_PATH = "h:\AI_Summarize\catwoman.txt_Ascii.txt"
CONTEXT_LENGTH=11989
OPENAI_API_BASE="http://localhost:1234/v1"
OPENAI_API_KEY="dummy_value"
MODEL_NAME="t-pro-it-1.0@q4_k_m"
# MODEL_NAME="t-pro-it-1.0@q5_k_m"


# Prompt templates (Provided in the prompt)
# reduce_prompt_template = """Ниже приведен набор резюме:
# {docs}
# Возьмите их и составьте окончательное, консолидированное резюме.
# РЕЗЮМЕ:"""
reduce_prompt_template = """Ниже приведен набор резюме:
# {docs}
# Возьмите их и составьте окончательное, консолидированное резюме.
# РЕЗЮМЕ:"""
reduce_prompt = PromptTemplate(template=reduce_prompt_template, input_variables=["docs"])

# map_prompt_template = """Ниже приведен набор документов
# {docs}
# Основываясь на этом списке документов, напишите краткое резюме.
# КРАТКОЕ РЕЗЮМЕ:"""
map_prompt_template = """Перескажи вкратце представленный отрывок:
{docs}"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["docs"])


set_llm_cache(SQLiteCache(database_path=".langchain.db"))

def detect_chunk_size_by_token_count(llm: ChatOpenAI, text: str, context_length: int) -> int:
    """
    Gets the token count for a given text using the LLM's API.
    """

    chunk_size=20000
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size
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

def summarize_with_map_reduce(
    llm: ChatOpenAI, docs: List[Document], map_prompt: PromptTemplate, reduce_prompt: PromptTemplate, context_length: int
):
    """
    Summarizes a list of documents using the map-reduce chain.

    Args:
        llm: The initialized LlamaCpp model.
        docs: A list of Document objects.
        map_prompt: The prompt for the map step.
        reduce_prompt: The prompt for the reduce step.

    Returns:
        The final summarized text.
    """
    try:

            # # This controls how each document will be formatted. Specifically,
            # # it will be passed to `format_document` - see that function for more
            # # details.
            # document_prompt = PromptTemplate(
            #     input_variables=["page_content"],
            #      template="{page_content}"
            # )
            # document_variable_name = "context"
            # llm = OpenAI()
            # # The prompt here should take as an input variable the
            # # `document_variable_name`
            # prompt = PromptTemplate.from_template(
            #     "Summarize this content: {context}"
            # )
            # llm_chain = LLMChain(llm=llm, prompt=prompt)
            # # We now define how to combine these summaries
            # reduce_prompt = PromptTemplate.from_template(
            #     "Combine these summaries: {context}"
            # )
            # reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt)
            # combine_documents_chain = StuffDocumentsChain(
            #     llm_chain=reduce_llm_chain,
            #     document_prompt=document_prompt,
            #     document_variable_name=document_variable_name
            # )
            # reduce_documents_chain = ReduceDocumentsChain(
            #     combine_documents_chain=combine_documents_chain,
            # )
            # chain = MapReduceDocumentsChain(
            #     llm_chain=llm_chain,
            #     reduce_documents_chain=reduce_documents_chain,
            # )
            # # If we wanted to, we could also pass in collapse_documents_chain
            # # which is specifically aimed at collapsing documents BEFORE
            # # the final call.
            # prompt = PromptTemplate.from_template(
            #     "Collapse this content: {context}"
            # )
            # llm_chain = LLMChain(llm=llm, prompt=prompt)
            # collapse_documents_chain = StuffDocumentsChain(
            #     llm_chain=llm_chain,
            #     document_prompt=document_prompt,
            #     document_variable_name=document_variable_name
            # )
            # reduce_documents_chain = ReduceDocumentsChain(
            #     combine_documents_chain=combine_documents_chain,
            #     collapse_documents_chain=collapse_documents_chain,
            # )
            # chain = MapReduceDocumentsChain(
            #     llm_chain=llm_chain,
            #     reduce_documents_chain=reduce_documents_chain,
            # )








        callbacks = [StdOutCallbackHandler()]
        # Define LLM Chains
        map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=False)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt, verbose=True, callbacks=callbacks)

        # Define MapReduceDocumentsChain
        combine_documents_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=ReduceDocumentsChain(
                combine_documents_chain=StuffDocumentsChain(
                    llm_chain=reduce_chain, document_variable_name="docs"
                ),
                token_max=context_length,  # Control the number of output tokens here
            ),
            document_variable_name="docs",
        )


        # Actually do the summarization
        intermediate_summaries = []
        #  Wrap the loop with tqdm for progress bar
        for doc in tqdm(docs, desc="Mapping chunks"):
            intermediate_summary = map_chain.run(doc.page_content)
            intermediate_summaries.append(intermediate_summary)

        # Print intermediate summaries and their sizes
        print("\nIntermediate Summaries:")
        for i, summary in enumerate(intermediate_summaries):
            # token_count = count_tokens(summary, tokenizer)
            print(f"  Chunk {i+1}: {summary[:100]}")  # Show first 100 chars

        # Combine intermediate summaries
        final_summary = combine_documents_chain.reduce_documents_chain.run(
            [Document(page_content=s) for s in intermediate_summaries]
        )
        return final_summary, intermediate_summaries

    except Exception as e:
        print(f"Error during summarization: {e}")
        return None


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


# Main Execution

if __name__ == "__main__":
    try:
        llm = ChatOpenAI(
            openai_api_base=OPENAI_API_BASE,
            openai_api_key=OPENAI_API_KEY,
            model_name=MODEL_NAME,
            temperature=0,
            # max_tokens=1000,  # Limit tokens for each response
            # max_retries=2,  # Retry on failure
        )

        text = ReadFile(FILE_PATH)

        # auto_chunk_size = detect_chunk_size_by_token_count(llm, text, context_length=CONTEXT_LENGTH)
        # docs = load_and_split_text(text, chunk_size=auto_chunk_size * 0.95)

        docs = split_text_with_token_limit(llm, text, CONTEXT_LENGTH)
          

        if docs:  # Only proceed if documents were loaded successfully

            final_summary, intermediate_summaries = summarize_with_map_reduce(
                llm, docs, map_prompt, reduce_prompt, CONTEXT_LENGTH
            )
            # final_summary, intermediate_summaries = summarize_with_map_reduce(llm, docs, map_prompt, reduce_prompt)

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