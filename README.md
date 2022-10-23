# The code of "LightEA: A Scalable, Robust, and Interpretable Entity Alignment Framework via Three-view Label Propagation"

## Datasets

* ent_ids_1: ids for entities in source KG;
* ent_ids_2: ids for entities in target KG;
* sup_ent_ids: training entity pairs;
* ref_ent_ids: testing entity pairs;
* triples_1: relation triples encoded by ids in source KG;
* triples_2: relation triples encoded by ids in target KG;
* graph_cache.pkl: cache file of graphs
* translated_ent_name: entity names translated by Google translator

Due to the limitation of upload file size, the DBP1M dataset is not included here.

## Environment

* Jupyter notebook
* tensorflow == 2.6.0
* Python == 3.7.0
* faiss
* Numpy
* tqdm
* pickle

## Just run LightEA.ipynb block by block.