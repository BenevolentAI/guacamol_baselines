import logging
import os
import pickle
import uuid
from abc import ABC, abstractmethod

from typing import Dict, Any

logger = logging.getLogger(__name__)


class FragStoreBase(ABC):
    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def add_records(self, collection: str, records):
        pass

    @abstractmethod
    def get_records(self, collection, query, return_count=False):
        pass

    @abstractmethod
    def save(self, path, collection=None):
        pass

    @abstractmethod
    def load(self):
        pass


class MemoryFragStore(FragStoreBase):
    """ Keep fragment database in memory """
    # todo include details of the fragmentation scheme in the load and save functions
    def __init__(self, path: str):
        self.store: Dict[str, Any] = {}
        super().__init__(path=path)

    def add_records(self, collection: str, records):
        """
        Add records to in_memory store.
        If provided a dict this method will overwrite any previous data with the same key (not currently used!)
        If provided an iterable, uuids will be generated and records stored in a dict
        """
        collection_records = self.store.get(collection, {})
        if not isinstance(records, dict):
            records = {uuid.uuid4(): row for row in records}
        collection_records.update(records)
        self.store[collection] = collection_records
        return True

    def get_records(self, collection, query, return_count=False):

        if collection not in self.store.keys():
            logger.warning(f"collection {collection} is not in fragstore")
            return []

        if collection == "genes":
            if len(query):
                raise NotImplementedError("genes store does not currently support custom queries")
            if return_count:
                return len(self.store[collection])
            else:
                return list(self.store[collection].values())

        if collection == "gene_types":
            if not len(query):
                return list(self.store[collection].keys())
            else:
                gt = self.store[collection].get(query["gene_type"], {})
                return (gt,) if len(gt) else []

    def save(self, path, collection=None):
        outdir = os.path.dirname(path)
        if outdir:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # save slim version for general usage (only gene_types key)
        self._save(path, "gene_types")

        # save all including parents (for debugging where frags came from)
        gene_type_outfile = path.replace(".pkl", "_with_genes.pkl")
        self._save(gene_type_outfile)
        logger.info(f"Saving: {gene_type_outfile}")

    def _save(self, path, collection=None):
        """ Save self.store to pickle, if collection_name is provided only save this collection. """
        with open(path, "wb") as f:
            if collection is None:
                pickle.dump(self.store, f)
            else:
                single_collection = {collection: self.store[collection]}
                pickle.dump(single_collection, f)

    def load(self):
        with open(self.path, "rb") as f:
            self.store: Dict[str, Any] = pickle.load(f)
            logger.info(f"loaded {self.path}")

        if "gene_types" in self.store:
            slim = {x["gene_type"]: x for x in self.store["gene_types"].values()}
            self.store["gene_types"] = slim


def fragstore_factory(frag_store_type: str, path: str):
    frag_stores = {
        "in_memory": MemoryFragStore,
    }
    if frag_store_type.lower() in frag_stores:
        return frag_stores[frag_store_type](path)
    else:
        raise NotImplementedError(f"frag store {frag_store_type} not recognised. "
                                  f"Valid frag stores: {list(frag_stores.keys())}")
