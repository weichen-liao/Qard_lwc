# Qard_lwc
This is the case study project of Qard

Objective: With the attached PDF (test1.pdf), implement a way to extract tokens representing companies out of it. We want to use spark with pyspark (on the local machine) for the execution framework

For Exploration:

If you wish to see the result, check Qard_Case_Study_SparkOCR+SpacyNER_CamembertNER.ipynb, the implementation is there

In Qard_Case_Study_test_NER.ipynb, you can find some exploration on multiple techniques that I tried. Some of them are not chosen to be used in the implementation.

For Deployment:

1. prepare the necessary lisence for spark-ocr: https://my.johnsnowlabs.com/
2. install the dependencies in requirements.txt
3. run pipeline.py
   this script will load all the PDFs under a given folder, and generate a json file for each PDF, where stores the extracted ORG entities from Spacy or Camembert
   
