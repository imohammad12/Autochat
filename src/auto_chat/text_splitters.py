import cleantext
import nltk
import copy
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Any, Iterable, Union, Optional, Dict
from .utils import add_cleaning_and_filtering_chunks
from .arguments import CompoundTextSplitterArguments, SPLITTER_TYPES
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter, TextSplitter


@add_cleaning_and_filtering_chunks
class RecursiveCharacterTextSplitterAndFilterer(RecursiveCharacterTextSplitter):
    pass


@add_cleaning_and_filtering_chunks
class HTMLHeaderTextSplitterAndFilterer(HTMLHeaderTextSplitter, TextSplitter):

    def create_documents(
            self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(tqdm(texts, desc="Splitting pages with HTML splitter")):
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                new_doc = Document(page_content=chunk.page_content, metadata={**metadata, **chunk.metadata})
                documents.append(new_doc)
        return documents


SPLITTER_TYPE_CLASS_MAP = {
    "html_splitter": HTMLHeaderTextSplitterAndFilterer,
    "recursive_char_splitter": RecursiveCharacterTextSplitterAndFilterer
}


class CompoundTextSplitter(ABC):
    """A text splitter class that combines multiple text splitters sequentially"""

    def __init__(self, args: CompoundTextSplitterArguments):
        self.splitters: Dict[SPLITTER_TYPES, TextSplitter] = {}
        self.args = args

        for splitter_type, splitter_args in self.args.splitters_args.items():
            splitter_class = SPLITTER_TYPE_CLASS_MAP[splitter_type]
            self.splitters[splitter_type] = splitter_class(
                min_words_drop=splitter_args.min_words_drop,
                do_clean=splitter_args.do_clean,
                **splitter_args.parent_text_splitter_kwargs
            )

    # The logic for how to use multiple splitters should be implemented in this method
    @abstractmethod
    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        pass


class HTMLAndRecursiveSplitter(CompoundTextSplitter):
    """A CompoundTextSplitter subclass that splits the pages first using the HTMLHeaderTextSplitterAndFilterer.
        Then the pages that did not have an HTML expected format are split using RecursiveCharacterTextSplitterAndFilterer
    """

    def __init__(self, args: CompoundTextSplitterArguments):
        super().__init__(args)
        for splitter in ['html_splitter', 'recursive_char_splitter']:
            assert splitter in self.splitters.keys(), (f"Splitter {splitter} not provided! "
                                                       f"HTMLAndRecursiveSplitter needs a {splitter} text splitter!")

    def split_documents(self, raw_documents: Iterable[Document]) -> List[Document]:
        html_splitter = self.splitters['html_splitter']
        recursive_splitter = self.splitters['recursive_char_splitter']
        html_parsed_splitted_docs = html_splitter.split_documents(raw_documents)
        not_splitted_docs, splitted_docs_with_headers = self.classify_docs(html_parsed_splitted_docs)
        print(f"Num of chunks splitted with HTML Headers: {len(splitted_docs_with_headers)}")
        print(f"Number of docs HTML Headers not found and passed "
              f"to Recursive Character Text Splitter: {len(not_splitted_docs)}")
        recursively_splitted_docs = recursive_splitter.split_documents(not_splitted_docs)
        all_document = recursively_splitted_docs + splitted_docs_with_headers
        return all_document

    @staticmethod
    def classify_docs(documents):
        """Determine which document is split by headers and which is not"""

        not_splitted_docs = []
        splitted_docs = []
        for idoc in documents:
            have_header = []
            for header in ["Header 1", "Header 2", "Header 3"]:
                have_header.append(header in idoc.metadata)
            if not any(have_header):
                not_splitted_docs.append(idoc)
            else:
                splitted_docs.append(idoc)
        return not_splitted_docs, splitted_docs
