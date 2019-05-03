from setuptools import setup, find_packages

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="chinese_gpt",
    version="0.1.0",
    author="Qingyang Wu",
    author_email="wilwu@ucdavis.edu",
    description="Chinese Generative Pre-Training Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qywu/Chinese-GPT",
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)