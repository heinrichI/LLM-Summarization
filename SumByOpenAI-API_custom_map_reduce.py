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
from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

from FileIO import save_summarization_results, ReadFile
from LmStudioUtils import get_context_length_from_lm_studio


FILE_PATH = "i:\Read\Художественное\Alias (8 book series)\[Alias Prequel 01] • Recruited (Mason, Lynn).epub"
OPENAI_API_BASE="http://localhost:1234/v1"
OPENAI_API_KEY="dummy_value"
# MODEL_NAME="t-pro-it-2.1"
MODEL_NAME="t-pro-it-1.0"
# MODEL_NAME="gemma-3-27b-it"
CONTEXT_LENGTH=0

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


def split_text_with_token_limit(llm: ChatOpenAI, text: str, context_length: int, overlap: int = 100, prompt: PromptTemplate = None):
    """Splits text into chunks respecting the model's token limit.
    
    Accounts for prompt template overhead so that the full prompt (template + 
    content + chat message formatting) stays within the model's context window.
    """
    # Estimate prompt template overhead: template tokens + ~20 tokens for chat wrapping
    template_overhead = 0
    if prompt is not None:
        template_without_docs = prompt.template.replace("{docs}", "").replace("  ", " ").strip()
        template_overhead = llm.get_num_tokens(template_without_docs) + 20  # +20 for chat message wrapping
    
    effective_limit = context_length - template_overhead
    safety_margin = 0.85  # Conservative margin to avoid edge cases
    max_chunk_tokens = int(effective_limit * safety_margin)
    
    print(f"Context: {context_length}, template_overhead: {template_overhead}, max_chunk_tokens: {max_chunk_tokens}")
    
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
        if llm.get_num_tokens(current_chunk + piece) <= max_chunk_tokens:
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
        # Map chain (one chunk at a time, so room for template tokens is needed)
        # Use a generous safety margin: chunk must not exceed 80% of context to leave room for prompt template
        map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=False, callbacks=[LoggingHandler()])

        # Reduce chain (combines batches of intermediate summaries)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt, verbose=True, callbacks=callbacks)

        # Final consolidation chain - wrap in StuffDocumentsChain for token-aware batching
        final_chain = LLMChain(llm=llm, prompt=final_prompt, verbose=True, callbacks=callbacks)
        final_combine_chain = StuffDocumentsChain(
            llm_chain=final_chain, document_variable_name="docs"
        )

        # Intermediate reduce chain (also wrapped for token-aware batching)
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        # --- MAP STEP ---
        intermediate_summaries = []
        for doc in tqdm(docs, desc="Mapping chunks"):
            intermediate_summary = map_chain.invoke(doc.page_content)
            intermediate_summaries.append(intermediate_summary['text'])

        # Print intermediate summaries and their sizes
        print("\nIntermediate Summaries:")
        for i, summary in enumerate(intermediate_summaries):
            print(f"  Chunk {i+1}: {summary[:100]}")

        # --- RECURSIVE REDUCE / FINAL STEP ---
        # Convert all intermediate summaries to Documents for chain processing
        current_batch = [Document(page_content=s) for s in intermediate_summaries]

        # Keep reducing in rounds until everything fits in one final pass
        # Estimate overhead for reduce/final prompt templates (whichever is larger)
        reduce_template_overhead = llm.get_num_tokens(
            reduce_prompt.template.replace("{docs}", "").replace("  ", " ").strip()
        ) + 20
        reduce_effective_limit = context_length - reduce_template_overhead
        reduce_batch_margin = 0.85
        
        while len(current_batch) > 1:
            combined_tokens = llm.get_num_tokens(
                "\n\n".join(d.page_content for d in current_batch)
            )
            if combined_tokens <= reduce_effective_limit * 0.85:
                break  # Fits in context, do one final consolidation

            print(f"\nIntermediate summaries total tokens {combined_tokens} > context. Reducing batch...")
            next_batch = []
            batch_buffer = []
            batch_buffer_tokens = 0
            max_batch_tokens = int(reduce_effective_limit * reduce_batch_margin)

            for doc in tqdm(current_batch, desc="Reducing summaries"):
                doc_tokens = llm.get_num_tokens(doc.page_content)
                # Check if adding this doc would exceed context (with safety margin)
                if batch_buffer and (batch_buffer_tokens + doc_tokens > max_batch_tokens):
                    # Reduce the current batch
                    reduced = combine_documents_chain.run(batch_buffer)
                    next_batch.append(Document(page_content=reduced))
                    batch_buffer = [doc]
                    batch_buffer_tokens = doc_tokens
                else:
                    batch_buffer.append(doc)
                    batch_buffer_tokens += doc_tokens

            # Don't forget the last batch
            if batch_buffer:
                reduced = combine_documents_chain.run(batch_buffer)
                next_batch.append(Document(page_content=reduced))

            current_batch = next_batch
            print(f"Reduced to {len(current_batch)} summaries")

        # Final consolidation
        final_summary = final_combine_chain.run(current_batch)
        return final_summary, intermediate_summaries

    except Exception as e:
        print(f"Error during summarization: {e}")
        return None, None


# Main Execution

if __name__ == "__main__":
    try:
        # Extract base URL (strip /v1 suffix if present)
        api_base = OPENAI_API_BASE.replace("/v1", "").rstrip("/")
        # Extract model key from MODEL_NAME (remove quantization suffix)
        model_key = MODEL_NAME.split("@")[0]

        # Get context_length dynamically from LM Studio API
        CONTEXT_LENGTH = get_context_length_from_lm_studio(api_base, model_key)

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

        docs = split_text_with_token_limit(llm, text, CONTEXT_LENGTH, prompt=map_prompt)
          

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
                save_summarization_results(FILE_PATH, final_summary, intermediate_summaries, MODEL_NAME)
            else:
                print("Summarization failed.")

        else:
            print("No documents to summarize. Check the input file and loading process.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")