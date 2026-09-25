<img width="4000" height="4618" alt="Untitled_Artwork" src="https://github.com/user-attachments/assets/0261ae9e-15ed-4840-9ba8-fbe69810800f" />

BioGeoFormer is a protein language model designed to predict and classify microbial proteins involved in key biogeochemical cycles — including methane, sulfur, nitrogen, and phosphorus transformations.
Built on the ESM-2 transformer architecture, fine-tuned on curated metabolic pathway databases (MCycDB, NCycDB, PCycDB, SCycDB), and a calibrated confidence function, BioGeoFormer extends sequence-function inference beyond traditional homology-based tools. 

Built on four databases, BioGeoFormer leverages 610 unique gene families to cover 36 metabolic pathways. It represents an excellent complementary method with classic approaches for metagenome and genome mining, to uncover hypothetical gene function related to biogeochemical cycling. 'BioGeoFormer' is a blanket term for the 8 fine-tuned models defined by their clustered identity splits, with training, validation, and test n% dissimilar at 10% intervals from 10% to 90%. While the nuance is described in our manuscript, we found that the 50% split model is the most effective at precisely identifying remote homologues, and recommend its use in most circumstances. 

While the tool does run on CPU-based infrastructure, we strongly recommend using a GPU-based infrastructure to annotate sequences to ensure the fastest completion time, especially for large datasets. If you do not have one personally available to you or do not have access through your institution, Google Colab is a user-friendly option to run a notebook with a GPU. 

### Current version

Version 1.0.0

### Using HuggingFace
```python
# Example: use BGF that was trained/evaluated on 10% GraphPart split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_ckpt = "nazbijari/bgf-i10"
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, trust_remote_code=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
encoding = tokenizer("MKTLLLTLLVVTIVCLDLGYSLKCYQHGKVVTCHRD", return_tensors="pt")
out = model(**encoding)

print(model.config.id2label[out.logits.argmax(-1).item()]) # to get predicted cycle
```
### Models available through HuggingFace
| Model | Split | Link |
| --- | --- | --- |
| `nazbijari/bgf-i10` | GraphPart 10% | https://huggingface.co/nazbijari/bgf-i10 |
| `nazbijari/bgf-i20` | GraphPart 20% | https://huggingface.co/nazbijari/bgf-i20 |
| `nazbijari/bgf-i30` | GraphPart 30% | https://huggingface.co/nazbijari/bgf-i30 |
| `nazbijari/bgf-i40` | GraphPart 40% | https://huggingface.co/nazbijari/bgf-i40 |
| `nazbijari/bgf-i50` | GraphPart 50% | https://huggingface.co/nazbijari/bgf-i50 |
| `nazbijari/bgf-i60` | GraphPart 60% | https://huggingface.co/nazbijari/bgf-i60 |
| `nazbijari/bgf-i70` | GraphPart 70% | https://huggingface.co/nazbijari/bgf-i70 |
| `nazbijari/bgf-i80` | GraphPart 80% | https://huggingface.co/nazbijari/bgf-i80 |
| `nazbijari/bgf-i90` | GraphPart 90% | https://huggingface.co/nazbijari/bgf-i90 |


### Preprint: 
https://www.biorxiv.org/content/10.64898/2025.12.17.695047v1

### Contact
Nima Azbijari: azbijarn@oregonstate.edu

Jacob Wynne: jacobwynne@ucsb.edu

### License
BioGeoFormer is under the MIT license.


