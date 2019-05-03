# Chinese-GPT 中文GPT预训练模型

Chinese Generative Pre-Training(GPT) Language Model

This project is unidirectional transformer GPT model (117M) trained on a large corpus dataset following the approach [OpenAI GPT-2](https://openai.com/blog/better-language-models/). Due to limited computational resources, we did not train our model from scratch. Instead, we take the advantage of [BERT](https://arxiv.org/abs/1810.04805) and use its weights as initialization to train our Chinese GPT. This makes the training possible on 4 x 1080Ti.

However, please notice that currently the performance still cannot match the original English GPT-2 model for various reasons. This can be that OpenAI has done better text filtering and has a dataset with better quality. Also, they have trained their model for about 300 GPU days at least. But the model here can be a good starting point if you want to apply it for substream tasks. 

## Installation 
Before using it, you might want to install the requirements first.

   ```bash
   pip install -r requirements.txt
   ```

You can also install it via `pip`.

   ```bash
   pip install chinese-gpt
   ```
   
## Usage

Check [tutorials](https://github.com/qywu/Chinese-GPT/tutorials) for details.

## Data Preparation
To train GPT, it requires a dataset from a wide range of sources.

We collected data from [NLP Chinese Corpus](https://github.com/brightmart/nlp_chinese_corpus)

In details, we used:

- 社区问答json版(webtext2019zh) ：大规模高质量数据集
- 百科类问答json版(baike2018qa)
- 新闻语料json版(news2016zh)

### Text Filtering

One thing to take care of is that text filtering. Since Bert Chinese tokenizer doesn't include some punctuations. You might want to use the following code to clean your data first:

```python
import regex as re

def filterPunctuation(x):
    x = re.sub(r'[‘’]', "'", x)
    x = re.sub(r'[“”]', '"', x)
    x = re.sub(r'[…]', '...', x)
    x = re.sub(r'[—]', '-', x)
    x = re.sub(r"&nbsp", "", x)
    return x
```

You may also want to convert traditional Chinese to simplified Chinese and apply some other filtering techniques based on your data. 

## Reference
 [OpenAI GPT-2](https://openai.com/blog/better-language-models/)
 
 [BERT](https://arxiv.org/abs/1810.04805)
 
 [AllenNLP](https://github.com/allenai/allennlp/)
 
 [Pytorch BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
 
 [NLP Chinese Corpus](https://github.com/brightmart/nlp_chinese_corpus)
