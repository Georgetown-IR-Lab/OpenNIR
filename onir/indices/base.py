from typing import Iterator
from onir import indices


class BaseIndex:
    def built(self) -> bool:
        raise NotImplementedError

    def build(self, doc_iter: Iterator[indices.RawDoc]):
        raise NotImplementedError

    def num_docs(self) -> bool:
        raise NotImplementedError

    def docids(self) -> Iterator[str]:
        raise NotImplementedError

    def get_raw(self, did: str) -> str:
        raise NotImplementedError
