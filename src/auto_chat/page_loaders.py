import os

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from typing import Union, Optional, Dict, List, Iterator, Any
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings



def get_bs_metadata(soup: Any) -> dict:
    """Build metadata from BeautifulSoup output."""
    metadata = {}
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    if description := soup.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", "No description found.")
    if html := soup.find("html"):
        metadata["language"] = html.get("lang", "No language found.")
    return metadata


class PageLoader(BaseLoader):
    """Parse HTML pages with BeautifulSoup'."""

    def __init__(
        self,
        pages_dir: str,
        autoset_encoding: bool = True,
        encoding: Optional[str] = None,
        default_parser: str = "html.parser",
        raise_for_status: bool = False,
        bs_get_text_kwargs: Optional[Dict[str, Any]] = None,
        bs_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize loader.

        Args:
            pages_dir: Path to the directory containing all html pages files.
            default_parser: Default parser to use for BeautifulSoup.
            raise_for_status: Raise an exception if http status code denotes an error.
            bs_get_text_kwargs: kwargs for beatifulsoup4 get_text
            bs_kwargs: kwargs for beatifulsoup4 web page parsing
        """
        self.pages_dir = pages_dir
        self.default_parser = default_parser
        self.raise_for_status = raise_for_status
        self.bs_get_text_kwargs = bs_get_text_kwargs or {}
        self.bs_kwargs = bs_kwargs or {}
        self.autoset_encoding = autoset_encoding
        self.encoding = encoding
        self.pages: List[Dict[str, str]] = self.load_html_pages()

    def load_html_pages(self) -> List[Dict[str, str]]:
        pages = []
        for page_name in os.listdir(self.pages_dir):
            page_path = os.path.join(self.pages_dir, page_name)
            name_splitted = page_name.split('.')
            page_format = name_splitted[-1]
            page_id = "".join(name_splitted[:-1])
            with open(page_path) as f:
                pages.append({
                    "raw_content": f.read(),
                    "id": page_id,
                    "format": page_format
                })

        return pages
    @staticmethod
    def _check_parser(parser: str) -> None:
        """Check that parser is valid for bs4."""
        valid_parsers = ["html.parser", "lxml", "xml", "lxml-xml", "html5lib"]
        if parser not in valid_parsers:
            raise ValueError(
                "`parser` must be one of " + ", ".join(valid_parsers) + "."
            )

    def parse(
        self,
        page: Dict[str, str],
        parser: Union[str, None] = None,
        bs_kwargs: Optional[dict] = None,
    ) -> Any:

        if parser is None:
            if page["format"] == "xml":
                parser = "xml"
            elif page["format"] == "html":
                parser = "html.parser"
            else:
                parser = self.default_parser

        self._check_parser(parser)

        return BeautifulSoup(page["raw_content"], parser, **(bs_kwargs or {}))


    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from HTML pages"""
        for page in self.pages:
            soup = self.parse(page, bs_kwargs=self.bs_kwargs)
            page_text = soup.get_text(**self.bs_get_text_kwargs)
            metadata = get_bs_metadata(soup)
            metadata["format"] = page["format"]
            metadata["id"] = page["id"]
            yield Document(page_content=page_text, metadata=metadata)


    def load_raw_pages(self) -> Iterator[Document]:
        """Loading pages as Documents object without parsing the pages with BeautifulSoup"""
        for page in self.pages:
            metadata = {"format": page["format"], "id": page["id"]}
            yield Document(page_content=page["raw_content"], metadata=metadata)

