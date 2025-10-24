import cv2
import numpy as np
import pandas as pd
import pytesseract
from tqdm import tqdm
import zipfile
import os
from PIL import Image
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity

def Get_siglip_embedding(text):
    # retrives the siglip embedding for a given text from Itau-group2/data/embeddings/siglip_glyphs/image_embeddings_test.npz

    embedding_path = "./data/embeddings/siglip_glyphs/image_embeddings_test.npz"
    embeddings = np.load(embedding_path, allow_pickle=True)
    vector_name = f"{text}.png"
    try:
        return embeddings[vector_name]
    except KeyError:
        print(f"Embedding file not found at {embedding_path}")
        return None
    
def Get_OCR_text(text):
    # uses pytesseract to extract text from an image located at image_path

    images_path = "./data/images/test_images.zip"
    image_name = f"{text}.png"
    try:
        with zipfile.ZipFile(images_path, 'r') as zf:
            new_image_name = zf.extract(image_name, "./data/images/")
    except FileNotFoundError:
        print(f"Image file not found at {images_path}")
        return None
    image = Image.open(new_image_name)
    image_gray = np.asarray(image.convert('L'))
    ocr_text = pytesseract.image_to_string(image_gray)
    os.remove(new_image_name)
    return ocr_text

def levenshtein_similarity(str1, str2):
    # computes the Levenshtein similarity between two strings

    lev_distance = levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    return 1 - lev_distance / max_len

def cosine_sim(vec1, vec2):
    # computes the cosine similarity between two vectors

    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

def run_SIGLIP_baseline(pair, thresh=0.8311601877212524):
    # runs the SIGLIP + cosine similarity baseline on a given pair of texts
    # default threshold is 0.8311601877212524

    text1, text2 = pair
    vec1 = Get_siglip_embedding(text1)
    vec2 = Get_siglip_embedding(text2)
    if vec1 is None or vec2 is None:
        print(f"Could not retrieve embeddings for pair {pair}")
        return None
    sim_score = cosine_sim(vec1, vec2)
    if sim_score >= thresh:
        print(f"Pair {pair} is classified as FRAUDULENT with confidence {sim_score}")
        confidence = (sim_score - thresh) / (1 - thresh)
    else:
        print(f"Pair {pair} is classified as NON-FRAUDULENT with confidence {sim_score}")
        confidence = (thresh - sim_score) / thresh
    return confidence

def run_OCR_levenshtein_baseline(pair, thresh=0.666667):
    # runs the OCR + Levenshtein similarity baseline on a given pair of texts
    # default threshold is 0.666667

    text1, text2 = pair
    ocr_text1 = Get_OCR_text(text1)
    ocr_text2 = Get_OCR_text(text2)
    if ocr_text1 is None or ocr_text2 is None:
        print(f"Could not retrieve OCR text for pair {pair}")
        return None
    sim_score = levenshtein_similarity(ocr_text1, ocr_text2)
    if sim_score >= thresh:
        print(f"Pair {pair} is classified as FRAUDULENT with confidence {sim_score}")
        confidence = (sim_score - thresh) / (1 - thresh)
    else:
        print(f"Pair {pair} is classified as NON-FRAUDULENT with confidence {sim_score}")
        confidence = (thresh - sim_score) / thresh
    return confidence

if __name__ == "__main__":

    pair1 = ("gizcoŧ", "gizbot") # fraudulent pair
    pair2 = ("epsb", "heypub") # non-fraud pair    

    # use siglip + consine similarity to score pairs
    run_SIGLIP_baseline(pair1)
    run_SIGLIP_baseline(pair2)

    # use OCR + levenshtein to score pairs
    run_OCR_levenshtein_baseline(pair1)
    run_OCR_levenshtein_baseline(pair2)
    