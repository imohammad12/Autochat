import json
import os.path
import argparse
import torch
from dataclasses import asdict
from src.auto_chat.arguments import (
    EmbeddingModelArguments,
    PageLoaderArguments,
    VectorDBArguments,
    CompoundTextSplitterArguments
)
from src.auto_chat.page_loaders import PageLoader
from src.auto_chat.text_splitters import HTMLAndRecursiveSplitter
from src.auto_chat.utils import convert_docs_embedding_to_faiss_format, init_embeddings, add_input_args
from langchain_community.vectorstores import FAISS
from transformers import HfArgumentParser

def creat_vector_db():
    parser = argparse.ArgumentParser()
    hf_parser = HfArgumentParser([
        PageLoaderArguments,
        EmbeddingModelArguments,
        VectorDBArguments,
        CompoundTextSplitterArguments
    ])
    add_input_args(parser)
    config_file_path = parser.parse_args().config_file_path
    page_loader_args, embed_args, vector_db_args, compound_text_splitter_args = hf_parser.parse_json_file(config_file_path)
    page_loader_args: PageLoaderArguments
    embed_args: EmbeddingModelArguments
    vector_db_args: VectorDBArguments
    compound_text_splitter_args: CompoundTextSplitterArguments

    # Loading the page loader
    page_loader = PageLoader(**asdict(page_loader_args))

    # Loading the embedding model
    embedding_model = init_embeddings(embed_args=embed_args)

    # Parsing and Splitting
    print(f"Parsing and Splitting Pages...")
    splitter = HTMLAndRecursiveSplitter(args=compound_text_splitter_args)
    raw_docments = list(page_loader.load_raw_pages())
    chunks = splitter.split_documents(raw_docments)


    print("Creating Embeddings and Deduplicating chunks...")
    deduplicated_chunks, embeddings_mat = embedding_model.embed_docs_with_deduplication(chunks)
    text_and_embeddings, metadatas = convert_docs_embedding_to_faiss_format(deduplicated_chunks, embeddings_mat)
    vector = FAISS.from_embeddings(
        text_embeddings=text_and_embeddings,
        embedding=embedding_model,
        metadatas=metadatas
    )
    vector_path = os.path.join(vector_db_args.db_path, embed_args.embedding_model_name)
    vector.save_local(vector_path)
    print(f"Vector DB is saved at {vector_path}.")

if __name__ == '__main__':
    creat_vector_db()