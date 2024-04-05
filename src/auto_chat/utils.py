import cleantext
import nltk
from dataclasses import asdict
from argparse import ArgumentParser
from langchain_core.documents import Document
from transformers import HfArgumentParser
from src.auto_chat.arguments import EmbeddingModelArguments
from src.auto_chat.embeddings import (
    HFEmbeddingsWithDeduplication,
    OpenAIEmbeddingsWithDeduplication,
    OpenAIEmbeddings,
    DeduplicationMixin
)
from typing import List, Any, Iterable, Union, Optional, Tuple


def add_cleaning_and_filtering_chunks(original_splitter):
    """Decorator for adding cleaning and small chunk filtering functionalities to a langchain text splitters class"""
    orig_split_text = original_splitter.split_text
    orig_init = original_splitter.__init__
    orig_merge_splits = original_splitter._merge_splits if hasattr(original_splitter, "_merge_splits") else None

    # Writing an __init__ function for the splitter to have the min_words_drop att to the original splitter class
    def __init__(self, min_words_drop, do_clean, **kwargs: Any):
        orig_init(self, **kwargs)
        self.min_words_drop = min_words_drop
        self.do_clean = do_clean

    # Writing a new split_text function for the original class to clean and filter small chunks
    def split_text(self, text: str) -> Union[List[str], List[Document]]:
        # Cleaning the text
        if self.do_clean:
            text = self.get_clean_text(text)

        # Splitting the text into smaller chunks with the original splitter
        chunks: Union[List[str], List[Document]] = orig_split_text(self, text=text)
        filtered_text_or_docs = []

        # Filtering texts that are very small
        for chunk in chunks:
            if isinstance(chunk, Document):
                chunk_text = chunk.page_content
            else:
                chunk_text = chunk
            if len(nltk.word_tokenize(chunk_text)) > self.min_words_drop:
                filtered_text_or_docs.append(chunk)

        return filtered_text_or_docs

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        filtered_splits = []
        for chunk in splits:
            if len(nltk.word_tokenize(chunk)) > self.min_words_drop:
                filtered_splits.append(chunk)

        return orig_merge_splits(self, filtered_splits, separator)

    @staticmethod
    def get_clean_text(text: str):
        clean_text = cleantext.clean(
            text,
            fix_unicode=True,  # fix various unicode errors
            to_ascii=True,  # transliterate to closest ASCII representation
            lower=False,  # lowercase text
            no_line_breaks=False,
            # fully strip line breaks as opposed to only normalizing them
            no_urls=True,  # replace all URLs with a special token
            no_emails=True,  # replace all email addresses with a special token
            no_phone_numbers=True,  # replace all phone numbers with a special token
            no_numbers=False,  # replace all numbers with a special token
            no_digits=False,  # replace all digits with a special token
            no_currency_symbols=True,  # replace all currency symbols with a special token
            no_punct=False,  # remove punctuations
            replace_with_punct="",  # instead of removing punctuations you may replace them
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="<NUMBER>",
            replace_with_digit="0",
            replace_with_currency_symbol="",
            lang="en"
        )
        return clean_text

    if orig_merge_splits:
        original_splitter._merge_splits = _merge_splits
    original_splitter.__init__ = __init__
    original_splitter.split_text = split_text
    setattr(original_splitter, "get_clean_text", get_clean_text)

    return original_splitter


def convert_docs_embedding_to_faiss_format(documents: List[Document], embeddings: List[float]) -> (
        Tuple[Iterable[Tuple[str, List[float]]], Iterable[dict]]
):
    """Converts a list of documents and a list of corresponding embeddings to faiss input format"""

    assert len(documents) == len(embeddings), "The text embeddings and documents should have the same length"

    texts = [d.page_content for d in documents]
    metadatas = [d.metadata for d in documents]
    texts_and_embeddings = zip(texts, embeddings)

    return texts_and_embeddings, metadatas


def init_embeddings(embed_args: EmbeddingModelArguments) -> (
        Union)[HFEmbeddingsWithDeduplication, OpenAIEmbeddingsWithDeduplication]:
    print(f"Initializing Embedding model...")
    embedding_other_kwargs = embed_args.other_embedding_kwargs
    embedding_main_kwargs = asdict(embed_args)
    all_embedding_kwargs = {**embedding_main_kwargs, **embedding_other_kwargs}
    if embed_args.embedding_type == "huggingface":
        embedding_model = HFEmbeddingsWithDeduplication(**all_embedding_kwargs)
    elif embed_args.embedding_type == "openai":
        open_ai_kwargs = {}
        for key, val in all_embedding_kwargs.items():
            if key in OpenAIEmbeddings.__fields__.keys() or key in DeduplicationMixin.__fields__.keys():
                open_ai_kwargs[key] = val
        embedding_model = OpenAIEmbeddingsWithDeduplication(**open_ai_kwargs)
    else:
        raise ValueError(f"Embedding class {embed_args.embedding_type} not supported! ")

    return embedding_model


def add_input_args(parser: ArgumentParser):
    parser.add_argument(
        "--config_file_path",
        type=str,
        default=None,
        help="Path to the json file containing all the configuration arguments for crating VectorDB."
    )
