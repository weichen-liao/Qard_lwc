# -*- coding: utf-8 -*-
# Author: Weichen Liao
import os
import json

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

from typing import List
import spacy
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import fr_core_news_sm

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

LICENCE_FILE = 'spark_nlp_for_healthcare_spark_ocr_3346.json'
# where the pdfs are stored
PDF_DIR = './PDFs/'
# where the results are stored, a json file for each pdf
RES_DIR = './RES/'

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]


# load the Spark OCR lisence
def read_licence_key(licence_file_name: str):
    with open(licence_file_name) as f:
        license_keys = json.load(f)

    secret = license_keys['SPARK_OCR_SECRET']
    os.environ['SPARK_OCR_LICENSE'] = license_keys['SPARK_OCR_LICENSE']
    os.environ['JSL_OCR_LICENSE'] = license_keys['SPARK_OCR_LICENSE']
    version = secret.split("-")[0]
    print('Spark OCR Version:', version)
    return secret, version


nlp_spacy = fr_core_news_sm.load()

tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
model_camembert = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
nlp_tf = pipeline('ner', model=model_camembert, tokenizer=tokenizer, aggregation_strategy="simple")

def Spacy_NER(texts: List[str]):
    orgs = []
    doc = nlp_spacy('\n'.join(texts))
    if doc.ents:
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                orgs.append(ent.text)
    return '|'.join(orgs)

def Camembert_NER(texts: List[str]):
    orgs = []
    entities = nlp_tf('\n'.join(texts))
    for entity in entities:
        if entity['entity_group'] == 'ORG':
            orgs.append(entity['word'])
    return '|'.join(orgs)

# Converting functions to UDF
Spacy_NER_UDF = udf(lambda z: Spacy_NER(z),StringType())
Camembert_NER_UDF = udf(lambda z: Camembert_NER(z),StringType())

# Transform pdf to image
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

secret, version = read_licence_key(LICENCE_FILE)

# start spark session
spark = start(secret=secret)

# load all the PDFs in the folder
image_df = spark.read.format("binaryFile").load(PDF_DIR).cache()
df = pipeline.transform(image_df).cache()

df.withColumn("orgs_spacy", Spacy_NER_UDF(col("text"))).withColumn("orgs_camembert", Camembert_NER_UDF(col("text")))

# save the results
df = df.toPandas()
for index, row in df.iterrows():
    file_name = row['path'].split('/')[-1].rstrip('.pdf')
    orgs_spacy = row['orgs_spacy']
    orgs_camembert = row['orgs_camembert']
    ners = {
        'spacy_ner': orgs_spacy,
        'camembert_ner': orgs_camembert,
    }
    with open(RES_DIR+file_name+'.txt', 'w') as f:
        f.write(json.dumps(ners))
