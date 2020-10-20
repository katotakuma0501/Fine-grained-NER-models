# Fine-grained-NER-models
## Citation
```
@inproceedings{DBLP:conf/acl/KatoAOMSI20,
  author    = {Takuma Kato and
               Kaori Abe and
               Hiroki Ouchi and
               Shumpei Miyawaki and
               Jun Suzuki and
               Kentaro Inui},
  title     = {Embeddings of Label Components for Sequence Labeling: {A} Case Study
               of Fine-grained Named Entity Recognition},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational
               Linguistics: Student Research Workshop, {ACL} 2020, Online, July 5-10,
               2020},
  pages     = {222--229},
  year      = {2020},
  crossref  = {DBLP:conf/acl/2020-s},
  url       = {https://www.aclweb.org/anthology/2020.acl-srw.30/},
  timestamp = {Thu, 25 Jun 2020 16:12:00 +0200},
  biburl    = {https://dblp.org/rec/conf/acl/KatoAOMSI20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
## Requirments
python
torch
transformer
## Run
```
git clone 
mkdir output
python run_ner.py --data_dir=data/ --bert_model=bert-base-cased --task_name=ner --output_dir=output --max_seq_length=128 --do_train --num_train_epochs 20 --do_eval --warmup_proportion=0.1
```
