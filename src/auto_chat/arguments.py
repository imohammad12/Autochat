import os
import torch
from dataclasses import dataclass, field
from typing import Union, Optional, Dict, Any, Literal, List

SPLITTER_TYPES = Literal['html_splitter', 'recursive_char_splitter']


@dataclass
class PageLoaderArguments:
    pages_dir: str = field(
        metadata={
            "help": ""
        }
    )

    autoset_encoding: bool = field(
        default=True,
        metadata={
            "help": ""
        }
    )

    encoding: Optional[str] = field(
        default=None,
        metadata={
            "help": ""
        }
    )
    default_parser: str = field(
        default="html.parser",
        metadata={
            "help": ""
        }
    )

    raise_for_status: bool = field(
        default=False,
        metadata={
            "help": ""
        }
    )

    bs_get_text_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": ""
        }
    )

    bs_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": ""
        }
    )


@dataclass
class EmbeddingModelArguments:
    embedding_type: Literal['huggingface', 'openai'] = field(
        default="huggingface",
        metadata={
            "help": "The type of embedding model."
        }
    )
    embedding_model_name: str = field(
        default="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        metadata={
            "help": "The name of embedding model"
        }
    )

    deduplication_threshold: float = field(
        default=0.9,
        metadata={
            "help": "The deduplication threshold used for filtering duplicated documents."
                    "Documents that have higher than deduplication_threshold cosine similarity are filtered"
                    "and one of them is kept. The threshold should be between zero and one. "
        }
    )

    embedding_openai_key: str = field(
        default=None,
        metadata={
            "help": "The OpenAI Key to use."
        }
    )

    cache_folder: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to store models. Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."
            }
        )

    model_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Keyword arguments to pass to the huggingface model."
        }
    )

    encode_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help":"Keyword arguments to pass when calling the `encode` method of the model."
        }
    )

    multi_process: bool = field(
        default=False,
        metadata={
            "help": "Run encode() on multiple GPUs."
        }
    )

    show_progress: bool = field(
        default=True,
        metadata={
            "help": "Whether to show a progress bar."
        }
    )

    other_embedding_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Any other kwarg wanted to pass to the embedding class initialization not declared here. "
        }
    )

    model: Optional[str] = field(
        init=False,
        metadata={
            "help": "The name of embedding model used for openai models"
        }
    )

    model_name: Optional[str] = field(
        init=False,
        metadata={
            "help": "The name of embedding model used for huggingface models"
        }
    )

    def __post_init__(self):
        # Setting the correct att model name according to each embedding type class
        # This is done because of having the same initialization format for all embedding type classes.
        if self.embedding_type == "huggingface":
            self.model_name = self.embedding_model_name
            self.model = None
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_kwargs['device'] = device
        if self.embedding_type == "openai":
            self.model = self.embedding_model_name
            self.model_name = None
            self.model_kwargs = {}

        if self.embedding_type == "openai":
            if not self.embedding_openai_key and not os.environ.get('OPENAI_API_KEY'):
                raise ValueError("OpenAI key is not set!")
            os.environ['OPENAI_API_KEY'] = self.embedding_openai_key or os.environ.get('OPENAI_API_KEY')


@dataclass
class TextSplitterArguments:
    type: SPLITTER_TYPES = field(
        metadata={
            "help": "The type of text splitter we want to use. "
                   "Currently just html_parser and recursive chat splitter are supported."
        }
    )
    min_words_drop: int = field(
        default = 5,
        metadata={
            "help": "The minium number of words a chunk of text should have to not to be dropped."
        }
    )
    do_clean: bool = field(
        default=False,
        metadata={
            "help": "If the given text should be cleaned before splitting. "
                    "Do NOT use this option when using a class of type html_splitter"
                    "because the cleaner may change the html format of text the html_splitter expects to receive."
        }
    )
    parent_text_splitter_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Any kw arguments needed to be set for the parent text spitter classes."
        }
    )


@dataclass
class CompoundTextSplitterArguments:
    splitters_configs: List[Dict[str, Any]] = field(
        metadata={
            "help": "List of all Configs of all text splitters. "
        }
    )

    splitters_args: Dict[SPLITTER_TYPES, TextSplitterArguments] = field(
        init=False,
        metadata={
            "help": "A dict containing all text splitter argument objects."
                    "It is created using the 'splitter_configs' list."
        }
    )

    def __post_init__(self):

        # Initializing the splitters arguments
        self.splitters_args = {}
        for splitter_config in self.splitters_configs:
            splitter_type = splitter_config['type']
            self.splitters_args[splitter_type] = TextSplitterArguments(**splitter_config)


@dataclass
class VectorDBArguments:
    vector_db_type: str = "faiss"
    db_path: str = "./data/vector_db"


@dataclass
class PipelineArguments:
    openai_key: str = field(
        metadata={
            "help": "The OpenAI Key to use."
        }
    )

    vector_db_path: str= field(
        metadata={
            "help": "Path to the saved vector DBs directory."
        }
    )

    num_retrieved_chunks: int = field(
        default=20,
        metadata={
            "help": "The number of retried chunks using the IR tool which will be passed to the LLM."
        }
    )

    chat_llm: str = field(
        default="gpt-3.5-turbo-1106",
        metadata={
            "help": "The backbone OpenAI model used as the main LLM."
        }
    )

    document_separator: str = field(
        default='\n###\n',
        metadata={
            "help": "The separator used for separating chunks in the prompt."
        }
    )

    agen_executor_kwarg: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "The kwargs used to initialize the langchain AgentExecutor."
        }
    )

    retriever_search_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "The kwargs used for searching options of the retriever."
                    "'num_retrieved_chunks' overrides the 'k' key of this kwargs (if passed)."
        }
    )

    def __post_init__(self):
        os.environ['OPENAI_API_KEY'] = self.openai_key or os.environ.get('OPENAI_API_KEY')


@dataclass
class ServerArguments:
    server_ip: str = field(
        default="0.0.0.0",
        metadata={
            "help": "The ip of the server."
        }
    )

    server_port: int = field(
        default=8892,
        metadata={
            "help": "The port of the server."
        }
    )


