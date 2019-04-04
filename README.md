# Chinese-GPT
Chinese Generative Pre-Training Language Model

## Data Preparation
To train GPT, it requires a dataset from a wide range of sources.

**THUCNews**
**Wechat 公众号**

### Text Filtering
Filtering the text before traininig is important in training a good language model. We set some basic rules here for filtering the dataset.

1. Translate Traditional Chinese into Simplified Chinese first. (Reduce vocabulary size) 
2. Remove all the space. (Chinese does not have space for tokenization)
3. Convert punctuations ，‘ “ ； ： （ ）？ 【】『』！～ —— to , ' " ; : () ? \[\] {} ! ~ --
4. Remove text in parenthesis ()

## Byte Pair Encoding
We use BPE as our choice of encoding. However, in many cases, this is not enough for Chinese. We need combine BPE and other SOTA tokenizer such as **jieba** to get better performance and eliminate misunderstanding.

Here are also some rules for our BPE vocabulary:

1. 256 UTF-8 characters (This helps us to encode any sentences.)
2. Common Chinese Punctuations
3. Common Chinese Characters
4. Common Chinese Words
