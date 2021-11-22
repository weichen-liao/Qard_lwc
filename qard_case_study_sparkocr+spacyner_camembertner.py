# -*- coding: utf-8 -*-
"""Qard_Case_Study_SparkOCR+SpacyNER/CamembertNER.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KdXTENo8yRK92NypLfxQZUWFRbVHXA5r

# Description
In this notebook, you can find the spark implementation of extracting company entities(ORG) from pdf

This Google Colab can be found in https://github.com/weichen-liao/Qard_lwc/blob/main/Qard_Case_Study_SparkOCR%2BSpacyNER_CamembertNER.ipynb

The OCR is based on Spark-OCR: https://github.com/JohnSnowLabs/spark-ocr-workshop

Spark OCR can extract texts from PDF files with relatively good accuracy. However, it's not free to use.

The NER is tried on 2 methods: Spacy-FR and Camembert. Comparison is made on these 2 methods

Conclusions are given in the end of this notebook

### Upload the files: pdf & licence key.json
"""

from google.colab import files
uploaded = files.upload()

!ls

"""### Read licence key"""

import os
import json

with open('spark_nlp_for_healthcare_spark_ocr_3346.json') as f:
    license_keys = json.load(f)

secret = license_keys['SPARK_OCR_SECRET']
os.environ['SPARK_OCR_LICENSE'] = license_keys['SPARK_OCR_LICENSE']
os.environ['JSL_OCR_LICENSE'] = license_keys['SPARK_OCR_LICENSE']
version = secret.split("-")[0]
print ('Spark OCR Version:', version)

"""### Install Dependencies"""

# Install Java
!apt-get update
!apt-get install -y openjdk-8-jdk
!java -version

# Install pyspark, SparkOCR, and SparkNLP
!pip install --ignore-installed -q pyspark==2.4.4
# Insall Spark Ocr from pypi using secret
!python -m pip install --upgrade spark-ocr==$version  --extra-index-url https://pypi.johnsnowlabs.com/$secret
# or install from local path
# %pip install --user ../../python/dist/spark-ocr-[version].tar.gz
!pip install --ignore-installed -q spark-nlp==2.5.2

# install spacy
! pip install spacy
! python -m spacy download en_core_web_sm
! python -m spacy download fr_core_news_sm

# install transformer for camembert-ner NER
! pip install transformers
! pip install sentencepiece

"""### Import Libraries"""

import pandas as pd
import numpy as np
import os

#Pyspark Imports
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

# Necessary imports from Spark OCR library
from sparkocr import start
from sparkocr.transformers import *
from sparkocr.enums import *
from sparkocr.utils import display_image, to_pil_image
from sparkocr.metrics import score
import pkg_resources

# import sparknlp packages
from sparknlp.annotator import *
from sparknlp.base import *

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

"""### Construct the OCR pipeline"""

pdf_to_image = PdfToImage() \
            .setInputCol("content") \
            .setOutputCol("image_raw") \
            .setKeepInput(True)

# Transform image to the binary color model
binarizer = ImageBinarizer() \
            .setInputCol("image_raw") \
            .setOutputCol("image") \
            .setThreshold(130)
# Run OCR for each region
ocr = ImageToText() \
            .setInputCol("image") \
            .setOutputCol("text") \
            .setIgnoreResolution(False) \
            .setPageSegMode(PageSegmentationMode.SPARSE_TEXT) \
            .setConfidenceThreshold(60)

#Render text with positions to Pdf document.
textToPdf = TextToPdf() \
            .setInputCol("positions") \
            .setInputImage("image") \
            .setInputText("text") \
            .setOutputCol("pdf") \
            .setInputContent("content")
# OCR pipeline
pipeline = PipelineModel(stages=[
            pdf_to_image,
            binarizer,
            ocr,
            textToPdf
        ])

"""### Start Spark Session"""

spark = start(secret=secret)
spark

"""### Load the pdf"""

image_df = spark.read.format("binaryFile").load('test1.pdf').cache()
image_df.show()

"""### Run OCR pipeline on every page"""

result = pipeline.transform(image_df).cache()
result_arr = []
for r in result.distinct().collect():
  for page in r.text:
    result_arr.append(page)

"""### Spacy NER
A transition-based named entity recognition component. The entity recognizer identifies non-overlapping labelled spans of tokens. The transition-based algorithm used encodes certain assumptions that are effective for “traditional” named entity recognition tasks, but may not be a good fit for every span identification problem. 
"""

import spacy
from spacy import displacy
import fr_core_news_sm

def show_ents(doc):
  if doc.ents:
    for ent in doc.ents:
      if ent.label_ == 'ORG':
        print(ent.text, ent.label_)
        # print(' | '.join([ent.text, ent.label_, str(spacy.explain(ent.label_))]))

nlp = fr_core_news_sm.load()
for i, text in enumerate(result_arr):
  print('----------------------------', 'page', i, '----------------------------')
  doc = nlp(text)
  show_ents(doc)
  # displacy.render is beautiful shown in Google Colab, but not so in PDF of Github
  # displacy.render(doc, style='ent',jupyter=True, options={'ents': ['ORG', 'PRODUCT']})

"""### camembert-ner NER
[camembert-ner] is a NER model that was fine-tuned from camemBERT on wikiner-fr dataset. Model was trained on wikiner-fr dataset (~170 634 sentences). Model was validated on emails/chat data and overperformed other models on this type of data specifically. In particular the model seems to work better on entity that don't start with an upper case.
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")

nlp_tf = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

for i, text in enumerate(result_arr):
  print('----------------------------', 'page', i, '----------------------------')
  entities = nlp_tf(text) 
  for entity in entities:
    if entity['entity_group'] == 'ORG':
      print(entity)

"""# Conclusions

In the given 10 page PDF, the expected company to be found is TUR-MAN, which is found by camembert-ner. However, both NER methods are in low precision, as lots of False Negative samples are found. I can't say the predictions are satistying, but at least a way to extract company names out of PDFs under the Pyspark framework is presented.

# To improve

1. Spark-OCR may not the best option for OCR as it's not free, the combination of pdf2image and pytesseract could be a good choice
2. Text preprocessing could be introduced in order to make the extracted text cleaner, hence increase the NER accuracy.
3. Translating FR into EN might improve the NER, but increase the cost at the same time.
4. train customized NER model which is based on the similar PDFs could greatly improve the NER accuracy, but annotation is needed.
"""

