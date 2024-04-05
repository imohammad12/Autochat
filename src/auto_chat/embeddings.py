import numpy as np
import scipy.spatial as sp
from typing import List, Union, Tuple, Any
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.pydantic_v1 import Extra, BaseModel, Field
from langchain_community.embeddings import HuggingFaceEmbeddings


class DeduplicationMixin(BaseModel):
    """A mixing class for a subclass of langchain base Embedding class
     Removes embeddings that are more similar than a threshold and keeps one of them"""
    deduplication_threshold: float = Field(
        description="The deduplication threshold used in this class."
    )

    class Config:
        extra = Extra.ignore

    def embed_docs_with_deduplication(self, documents: List[Document]) -> Tuple[List[Document], List[float]]:
        """Removes duplicated documents and returns the documents with embeddings"""
        texts = [d.page_content for d in documents]

        # self: Embeddings
        embed_mat = self.embed_documents(texts=texts)
        keeping_indices = self.deduplicate(embed_mat)
        deduplicated_docs = [documents[i] for i in keeping_indices]
        deduplicated_embeddings = [embed_mat[i] for i in keeping_indices]
        assert len(deduplicated_embeddings) == len(deduplicated_docs)

        print(f"From {len(documents)} chucks, {len(documents) - len(deduplicated_docs)} duplicated chunks are removed.")

        return deduplicated_docs, deduplicated_embeddings

    def deduplicate(self, embed_mat: Union[List[List[float]], np.array]) -> List[int]:
        """returns a list of indices of embeddings that should be kept. The rest should be removed"""
        if isinstance(embed_mat, list):
            embed_mat = np.array(embed_mat)

        # Obtaining the similarity matrix
        sim_mat = 1 - sp.distance.cdist(embed_mat, embed_mat, 'cosine')
        deduplicated_indices = []
        for i in range(sim_mat.shape[0]):
            if i != sim_mat.shape[0] - 1 and np.max(sim_mat[i][i + 1:]) > self.deduplication_threshold:
                continue
            deduplicated_indices.append(i)

        return deduplicated_indices


class HFEmbeddingsWithDeduplication(HuggingFaceEmbeddings, DeduplicationMixin):
    """HuggingFaceEmbeddings with ignored extra kwargs instead of forbidden extra kwargs and deduplication"""

    class Config:
        extra = Extra.ignore


class OpenAIEmbeddingsWithDeduplication(OpenAIEmbeddings, DeduplicationMixin):
    class Config:
        extra = Extra.ignore


