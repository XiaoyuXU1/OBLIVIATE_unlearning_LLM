# generate translation documents
Please generate four similar novels based on the following document. Requirements: the same characters, places and events must not appear, and the number of words must be similar, answer in four times, each time answering one novel.


## Installation

```
conda create -n unlearn python=3.10.15
conda activate unlearn
conda install pytorch==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn==2.5.3 --no-build-isolation
pip install numpy==1.26.4
python -m spacy download en_core_web_sm
```