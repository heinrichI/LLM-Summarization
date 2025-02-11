import os
from typing import List

from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from tqdm import tqdm  # Import tqdm
from langchain_core.callbacks.stdout import StdOutCallbackHandler

# 1. Configuration and Setup

# Model path (Adjust this to your actual path)
MODEL_PATH = "h:/AI_Summarize/models/t-tech/T-pro-it-1.0-Q4_K_M-GGUF/t-pro-it-1.0-q4_k_m.gguf" # Use the correct filename

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Prompt templates (Provided in the prompt)
reduce_prompt_template = """Ниже приведен набор резюме:
{docs}
Возьмите их и составьте окончательное, консолидированное резюме.
РЕЗЮМЕ:"""
reduce_prompt = PromptTemplate(template=reduce_prompt_template, input_variables=["docs"])

map_prompt_template = """Ниже приведен набор документов
{docs}
Основываясь на этом списке документов, напишите краткое резюме.
КРАТКОЕ РЕЗЮМЕ:"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["docs"])


# 2. Initialize LLM (LlamaCpp)

def initialize_llm(model_path: str, n_ctx: int = 8096):
    """
    Initializes the LlamaCpp model.

    Args:
        model_path: The path to the GGUF model file.
        n_ctx: The context size for the model.

    Returns:
        An instance of the LlamaCpp model.
    """
    try:
        llm = LlamaCpp(
            model_path=model_path,
            n_ctx=n_ctx,
            temperature=0, 
            n_gpu_layers=-1, #  Change this value based on your GPU
            verbose=True # Set to True for debugging output,
            # max_tokens=8000
        )
        return llm
    except Exception as e:
        print(f"Error initializing LlamaCpp: {e}")
        raise

# 3. Load and Split the Text

def load_and_split_text(file_path: str, chunk_size: int = 20000, chunk_overlap: int = 100):
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
        with open(file_path, "r", encoding="utf-8") as f:
            document = f.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_text(document)[:1]
        print(f"Text splitted on: {len(texts)} pices")
        #  Convert text chunks into Langchain Document objects
        docs = [Document(page_content=t) for t in texts]
        return docs
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return []  # Return an empty list in case of an error
    except Exception as e:
        print(f"Error reading or splitting text: {e}")
        return []


# 4. Summarization using Map Reduce

def summarize_with_map_reduce(
    llm: LlamaCpp, docs: List[Document], map_prompt: PromptTemplate, reduce_prompt: PromptTemplate
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
        callbacks = [StdOutCallbackHandler()]
        # Define LLM Chains
        map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=True, callbacks=callbacks)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt, verbose=True, callbacks=callbacks)

        # Define StuffDocumentsChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
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
        final_summary = combine_documents_chain.run(
            [Document(page_content=s) for s in intermediate_summaries]
        )
        return final_summary, intermediate_summaries

    except Exception as e:
        print(f"Error during summarization: {e}")
        return None

def save_summarization_results(file_path: str, final_summary: str, intermediate_summaries: List[str]):
    """
    Saves the summarization results to files near the source text file.
    
    Args:
        file_path: Path to the original text file
        final_summary: The final summary text
        intermediate_summaries: List of intermediate summaries
    """
    try:
        # Get the directory and filename without extension
        file_dir = os.path.dirname(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(file_dir, f"{file_name}_summaries")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save final summary
        final_summary_path = os.path.join(results_dir, f"{file_name}_final_summary.txt")
        with open(final_summary_path, 'w', encoding='utf-8') as f:
            f.write(final_summary)
        
        # Save intermediate summaries
        intermediate_summary_path = os.path.join(results_dir, f"{file_name}_intermediate_summaries.txt")
        with open(intermediate_summary_path, 'w', encoding='utf-8') as f:
            for i, summary in enumerate(intermediate_summaries, 1):
                f.write(f"=== Intermediate Summary {i} ===\n")
                f.write(summary)
                f.write("\n\n")
        
        print(f"\nSummaries saved in: {results_dir}")
        print(f"Final summary: {final_summary_path}")
        print(f"Intermediate summaries: {intermediate_summary_path}")
        
    except Exception as e:
        print(f"Error saving summarization results: {e}")

# 5. Main Execution

if __name__ == "__main__":
    #  Example Usage:
    file_path = "h:/AI_Summarize/catwoman_UTF8.txt" # Replace with your file path. Ensure this file exists.

    try:
        llm = initialize_llm(MODEL_PATH)
        docs = load_and_split_text(file_path)


        if docs:  # Only proceed if documents were loaded successfully
            final_summary, intermediate_summaries = summarize_with_map_reduce(llm, docs, map_prompt, reduce_prompt)

            if final_summary:
                print("Summary:")
                print(final_summary)

                  # Save the results
                save_summarization_results(file_path, final_summary, intermediate_summaries)
            else:
                print("Summarization failed.")

        else:
            print("No documents to summarize. Check the input file and loading process.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")