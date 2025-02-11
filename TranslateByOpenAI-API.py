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

from FileIO import ReadFile



FILE_PATH = "f:\Miles to Go.epub"


CONTEXT_LENGTH=11989
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
# map_prompt_template = """Переведи на русский:
# {docs}"""
# map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["docs"])


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

def tranclate(
    llm, docs: List[Document], context_length: int
):
    """Summarizes with map-reduce, handling context overflow."""
    try:
        template = """Переведи без повторений следующий текст на русский язык:
        {docs}
        """
        prompt = PromptTemplate(template=template, input_variables=["docs"])
        # callbacks = [StdOutCallbackHandler()]
        # Define LLM Chains
        chain = LLMChain(llm=llm, prompt=prompt, verbose=False, callbacks=[LoggingHandler()])

        # Define StuffDocumentsChain
        # combine_documents_chain = StuffDocumentsChain(
            # llm_chain=reduce_chain, document_variable_name="docs"
        # ) 
        # final_chain2 = StuffDocumentsChain(
        #     llm_chain=final_chain, document_variable_name="docs"
        # )

        from langchain_core.messages import HumanMessage, SystemMessage


        intermediate_summaries = []
        for doc in tqdm(docs, desc="Mapping chunks"):
            # translated_chunk = chain.invoke(doc.page_content)

            messages = [
                SystemMessage(
                    content="Ты профессиональный переводчик. Переводи без повторов пользовительские предложения на русский язык."
                ),
                HumanMessage(
                    content=doc.page_content
                )
            ]
            translated_chunk = llm.invoke(messages)

            print(translated_chunk)
            intermediate_summaries.append(translated_chunk['text'])

        return "\n".join(intermediate_summaries)

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
            # temperature=0,
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

            tranclated_text = tranclate(
                llm, docs, CONTEXT_LENGTH
            )

            if tranclated_text:
                # Save the results
                file_dir = os.path.dirname(FILE_PATH)
                file_name = os.path.splitext(os.path.basename(FILE_PATH))[0]
                tranclated_path = os.path.join(file_dir, f"{file_name}_tranclated.txt")
                with open(tranclated_path, 'w', encoding='utf-8') as f:
                    f.write(text)
            else:
                print("Tranclated failed.")

        else:
            print("No documents to summarize. Check the input file and loading process.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")