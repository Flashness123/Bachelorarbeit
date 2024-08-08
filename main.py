#!/usr/bin/env python
# coding: utf-8
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


import pandas as pd
from fuzzywuzzy import process, fuzz
import warnings
import logging
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
import torch
import numpy as np
import os
import torch
from transformers import LayoutLMv3Processor, AutoModelForTokenClassification
from datasets.features import ClassLabel
import easyocr
import pytesseract
import gradio as gr
from typing import List
from datasets import load_dataset
import concurrent.futures

file_path = r'.\EPLASS Plannummernkonvention_20231016.xlsx'

def extract_parentheses(text):
    # Find text within the first set of parentheses
    result = re.search(r'\((.*?)\)', text)
    return result.group(1) if result else text

def format_with_leading_zeros(number, desired_length):
    # Convert the number to an integer if it's a float without a decimal part,
    # otherwise, leave it as is (it will be converted to a string next).
    if isinstance(number, float) and number.is_integer():
        number = int(number)

    # Convert the number to a string
    number_str = str(number)

    # Pad with leading zeros
    return number_str.zfill(desired_length)

def get_pure_dataset(answer):
    if pd.isna(answer):
        return answer  # Return NaN as is to avoid type errors

    answer = str(answer)
    # Define replacements for German umlauts and ss
    replacements = {
        'ö': 'o',
        'Ö': 'O',
        'ä': 'a',
        'Ä': 'A',
        'ü': 'u',
        'Ü': 'U',
        'ß': 'ss'
    }

    # Replace each German character with its replacement
    for german_char, replacement in replacements.items():
        answer = answer.replace(german_char, replacement)


    # Remove specific substrings
    answer = answer.replace("Deutschland", "").replace("deutschland", "").replace("GmbH", "").replace("gmbh", "")

    # Optionally, remove spaces and convert to lower case
    answer = re.sub(r'\s+', '', answer)  # Remove all kinds of whitespace
    answer = answer.lower()  # Convert to lower case for uniformity

    return answer

def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

def get_pure_answer(question:str, answer: str):
    answer = str(answer)
    while question in answer:
        answer = answer.replace(question, '')
    answer = re.sub(r'[,.\/\-=:;_ ]', '', answer)
    answer = answer.lower()
    answer = answer.replace("gmbh", "")
    answer = answer.replace("deutschland", "")
    answer = answer.replace("nr", "")# rausnehmen?

    replacements = {
        'ö': 'o',
        'ä': 'a',
        'ü': 'u',
        'ū': 'u',
        'ß': 'ss',
        'massstm': 'massst',

        'odos': 'eqos'
    }

    # Replace each German character with its replacement
    for german_char, replacement in replacements.items():
        answer = answer.replace(german_char, replacement)

    return answer

def split_image_horizontally(image: Image, max_tokens: int = 512) -> List[Image]:
    """
    Splits an image horizontally into smaller segments each with fewer than max_tokens tokens.

    Parameters:
        image (PIL.Image): The original image to be split.
        processor: The processor used for tokenizing.
        max_tokens (int): The maximum number of tokens for each image segment.

    Returns:
        List of Image objects, each representing a segment of the original image.
    """
    width, height = image.size
    segments = []
    current_height = 0

    # Continuously try to split until the entire height is covered
    while current_height < height:
        min_height = 100  # Minimum height increase to attempt finding a valid segment
        found = False
        last_valid_height = current_height + min_height  # Default to minimal segment in case of no valid split found

        for h in range(current_height + min_height, height + min_height, min_height):
            segment = image.crop((0, current_height, width, h))
            encoding = processor(segment, return_offsets_mapping=True, return_tensors="pt")
            num_tokens = encoding.input_ids.size(1)
            # print(f"Trying height {h}: {num_tokens} tokens")

            if num_tokens < max_tokens:
                last_valid_height = h
                found = True
            else:
                if found:
                    break  # Use the last valid height
                else:
                    # If even the smallest increment is too large, reduce min_height
                    min_height = max(50, min_height // 2)
                    break

        # Crop using the last valid height that had fewer than max_tokens
        segment = image.crop((0, current_height, width, last_valid_height))
        segments.append(segment)
        current_height = last_valid_height  # Move the current height to the end of the last valid segment

        if not found:
            # Adjust in case no valid segment is found to prevent infinite loop
            current_height += min_height

    return segments

def mast_extraction(image, image_name = "Current Image"):
    encoding = processor(image, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')

    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()
    width, height = image.size
    pred_boxes = [unnormalize_box(box, width, height) for box in token_boxes]
    pred_labels = [id2label[pred] for pred in predictions]

    #Do I need these?:
    # Filter out only the bounding boxes labeled "mast"
    mast_boxes = []
    # Filter out only the bounding boxes labeled "mast" and remove duplicates or similar boxes
    unique_mast_boxes = []
    seen = set()
    for box, label in zip(pred_boxes, pred_labels):
        if label == "B-Mast":
            box_tuple = tuple(box)
            if box_tuple not in seen:
                unique_mast_boxes.append(box)
                seen.add(box_tuple)

    # Cropping out the tokens that correspond with Mast and saving them in Mast to visualize
    cropped_images = crop_boxes_with_margin(image, unique_mast_boxes)
    for i, cropped_image in enumerate(cropped_images):
        directory_path = f".\data\mast_extract\{image_name}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        file_path = os.path.join(directory_path, f"cropped_mast_{i + 1}.png")
        cropped_image.save(file_path)

    extracted_texts = [extract_text_from_image_easy_ocr(cropped_image) for cropped_image in cropped_images]
    # Sort out empty strings
    merged_sentence = ' '.join(extracted_texts)
    words = merged_sentence.split()
    #print("extracted_texts:", extracted_texts)

    mast_index = None
    mast_bbox = None

    #finds first occurence of Mast or Nr
    for i, text in enumerate(extracted_texts):
        if "mast" in text.lower() or "nr" in text.lower() or "m" in text.lower():
            mast_index = i
            mast_bbox = unique_mast_boxes[i]
            break
    #print(f"mast_index: {mast_index}")

    mast_occurences = ""

    if mast_index is not None:
        while mast_index < len(extracted_texts):
            current_text = [w.strip() for w in extracted_texts[mast_index:]]
            #print(f"current_text: {current_text}")

            if not has_keyword_followed_by_digit(current_text, ["Mast", "Nr."]):
                counter = 0
                ocr_result = None
                mast_bbox = unique_mast_boxes[mast_index]
                #keine nummer zwischen den ersten beiden keywortern
                while counter < 1:
                    ocr_result = adjust_and_ocr(image, mast_bbox)

                    #print(f"counter: {counter}")
                    counter +=1

                    # Break if a number is found after adjustment
                    if any(char.isdigit() for char in ocr_result):
                        #print("Broken out")
                        mast_occurences += ocr_result
                        break
            else:
                mast_occurences += ', ' + ' '.join(current_text)
                mast_index += 1
            mast_index += 1
        print(f"mast_occurences: {mast_occurences}")
        return mast_occurences

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]
def crop_boxes_with_margin(image, boxes, margin=4):
    cropped_images = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.width, x2 + margin)
        y2 = min(image.height, y2 + margin)
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_images.append(cropped_image)
    return cropped_images

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def extract_text_from_image_easy_ocr(image):
    reader = easyocr.Reader(['en','de'])
    image_np = np.array(image)
    results = reader.readtext(image_np)
    full_text = ''
    for (bbox, text, prob) in results:
        full_text += text + " "
    full_text = full_text.strip()
    return full_text
#     return " ".join(text for (_, text, _) in reader.readtext(image)).strip()


def adjust_and_ocr(image, bbox, x_extend=130, y_adjust=2):
    x1, y1, x2, y2 = bbox
    x2 += x_extend
    y1 -= y_adjust
    y2 += y_adjust
    cropped_adjusted_image = image.crop((x1, y1, x2, y2))
    #cropped_adjusted_image.show()
    return extract_text_from_image_easy_ocr(cropped_adjusted_image)

def has_keyword_followed_by_digit(extracted_texts, keywords):
    positions = []  # To hold positions of keywords
    # Convert list of keywords to lowercase once to improve efficiency
    keywords = [kw.lower() for kw in keywords]

    # Iterate through the extracted texts to find keywords
    for index, text in enumerate(extracted_texts):
        text_lower = text.lower()
        if any(kw in text_lower for kw in keywords):
            positions.append((index, text_lower))

    # Check for digits between the first and second keyword occurrences
    if len(positions) >= 2:
        # Extract the segment of texts between the first and second keyword
        segment = extracted_texts[positions[0][0] + 1:positions[1][0]]
        # Check if any text in the segment contains a digit
        for text in segment:
            if any(char.isdigit() for char in text):
                return True

    # Handle cases where only one keyword is found and then the list ends
    elif len(positions) == 1:
        # Check remaining texts after the first keyword for any digits
        segment = extracted_texts[positions[0][0] + 1:]
        for text in segment:
            if any(char.isdigit() for char in text):
                return True

    return False  # Return False if no suitable pattern is found
def metadata_extraction(image):
    res = []
    donut_processor = DonutProcessor.from_pretrained("Resi/donut-docvqa-sagemaker")
    donut_model = VisionEncoderDecoderModel.from_pretrained("Resi/donut-docvqa-sagemaker")
    output_dict = {}

    # Your existing code
    prompts = ["Leitungsanlage", "Ersteller", "Dokumenttyp", "Massstab", "Mast"]
    pixel_values = donut_processor(image, return_tensors="pt").pixel_values

    def process_prompt(prompt):
        decoder_input_ids = donut_processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
            "input_ids"]

        outputs = donut_model.generate(pixel_values,
                                       decoder_input_ids=decoder_input_ids,
                                       max_length=donut_model.decoder.config.max_position_embeddings,
                                       early_stopping=False,
                                       pad_token_id=donut_processor.tokenizer.pad_token_id,
                                       eos_token_id=donut_processor.tokenizer.eos_token_id,
                                       use_cache=True,
                                       num_beams=1,
                                       bad_words_ids=[[donut_processor.tokenizer.unk_token_id]],
                                       return_dict_in_generate=True,
                                       output_scores=True)
        seq = donut_processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(donut_processor.tokenizer.eos_token, "").replace(donut_processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token

        scores = outputs.scores  # List of tensors with log probabilities
        probs = [torch.softmax(score, dim=-1) for score in scores]  # Convert to probabilities
        max_probs = [torch.max(prob, dim=-1)[0] for prob in probs]  # Get max probability for each step
        confidence_values = [prob.item() for prob in max_probs]  # Convert to list of floats
        average_confidence = np.mean(confidence_values)

        return prompt, {'sequence': get_pure_answer(prompt, seq), 'average_confidence': average_confidence}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_prompt, prompt): prompt for prompt in prompts}
        for future in concurrent.futures.as_completed(futures):
            prompt, result = future.result()
            output_dict[prompt] = result
            print(prompt, result['sequence'], result['average_confidence'])

    def process_key_value(key, value):
        res_local = []
        if key == 'Massstab':
            return res_local

        threshold = confidence_thresholds.get(key, 100.0) / 100.0

        if value['average_confidence'] > threshold:
            if key in column_mapping:
                column_name = column_mapping[key]
                if column_name in data.columns and column_name not in ['Dokumenttyp', 'Mast']:
                    cleansed_column = data[column_name].apply(lambda x: get_pure_answer(key, x))
                    result = process.extractOne(value['sequence'], cleansed_column)
                    if result:
                        best_match, match_score = result[0], result[1]
                        print(f"Nearest word for {column_name} based on {value['sequence']}: {best_match} with score {match_score}")
                        res_local.append(f"Nearest word for {column_name} based on {value['sequence']}: {best_match} with score {match_score}")
                    else:
                        print(f"No suitable match found for {value['sequence']} in column {column_name}")
                        res_local.append(f"No suitable match found for {value['sequence']} in column {column_name}")
                elif column_name == 'Dokumenttyp':
                    cleansed_column = data[column_name].apply(lambda x: get_pure_answer(key, x))
                    result = process.extractOne(value['sequence'], cleansed_column)
                    data[column_name] = data[column_name].fillna('')
                    if result:
                        best_match, match_score = result[0], result[1]
                        print(f"Nearest word for {column_name} based on {value['sequence']}: {best_match} with score {match_score}")
                        res_local.append(f"Nearest word for {column_name} based on {value['sequence']}: {best_match} with score {match_score}")
                    print("But could be: ")
                    res_local.append("But could be: ")
                    pattern = f"{value['sequence']}"
                    matches = data[cleansed_column.str.contains(pattern, regex=True, case=False)]
                    if not matches.empty:
                        for match in matches[column_name]:
                            print(f"Word in {column_name}: {match}")
                            res_local.append(f"Word in {column_name}: {match}")
                elif column_name == 'Mast':
                    cleansed_column = data[column_name].apply(lambda x: get_pure_answer(key, x))
                    result = process.extractOne(value['sequence'], cleansed_column)
                    if result:
                        best_match, match_score = result[0], result[1]
                        print(f"Nearest word for {column_name} based on {value['sequence']}: {best_match} with score {match_score}")
                        res_local.append(f"Nearest word for {column_name} based on {value['sequence']}: {best_match} with score {match_score}")
                    else:
                        print(f"No suitable match found for {value['sequence']} in column {column_name}")
                        res_local.append(f"No suitable match found for {value['sequence']} in column {column_name}")
                    print("LayoutLM output: ")
                    res_local.append("LayoutLM output: ")
                    segments = split_image_horizontally(image)
                    extracted_mast = ""
                    for i, seg_image in enumerate(segments):
                        extracted_mast += str(mast_extraction(seg_image))
                    print(extracted_mast)
                    res_local.append(extracted_mast)
                else:
                    print(f"Column {column_name} not found in DataFrame.")
                    res_local.append(f"Column {column_name} not found in DataFrame.")
            else:
                print(f"No column mapping found for key: {key}")
                res_local.append(f"No column mapping found for key: {key}")
        else:
            print(f"Not confident for {key}: confidence {value['average_confidence']}")
            res_local.append(f"Not confident for {key}: confidence {value['average_confidence']}")
        return res_local

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_key_value, key, value): key for key, value in output_dict.items()}
        for future in concurrent.futures.as_completed(futures):
            res.extend(future.result())

    layoutlm_output = ""
    other_results = []

    for item in res:
        if item.startswith("LayoutLM output:"):
            layoutlm_output = res[res.index(item) + 1]
        else:
            other_results.append(item)

    # Format the output using HTML
    formatted_output = "<div style='display: flex;'>"
    formatted_output += "<div style='flex: 1; padding-right: 10px;'>"
    formatted_output += "<h3>Results:</h3><ul>"

    for item in other_results:
        formatted_output += f"<li>{item}</li>"

    formatted_output += "</ul></div>"
    formatted_output += "<div style='flex: 1; padding-left: 10px;'>"
    formatted_output += "<h3>LayoutLM Output:</h3><pre>"
    formatted_output += layoutlm_output
    formatted_output += "</pre></div></div>"

    return formatted_output

data = pd.read_excel(file_path, skiprows=14)
new_columns = [
    "Projektkürzel_bezeichner", "Projektkürzel", "Leitungsanlage_bezeichner", "Leitungsanlage",
    "Mast_bezeichner", "Mast", "Erstellerschlüssel_bezeichner", "Erstellerschlüssel",
    "Dokumentenart_bezeichner", "Dokumentenart", "Dokumenttyp_bezeichner", "Dokumenttyp",
    "lfd. Nummer", "Index"
]
current_columns = data.columns
column_rename_map = dict(zip(current_columns, new_columns))
data.rename(columns=column_rename_map, inplace=True)
pd.set_option('display.max_rows', None)  # Adjust based on your preference
pd.set_option('display.max_columns', None)  # Adjust to display all columns
pd.set_option('display.width', 1000)  # Adjust the width to fit your screen
pd.set_option('display.colheader_justify', 'center')  # Center the column headers
pd.set_option('display.precision', 3)  # Set the precision of floating point numbers
data.columns = [extract_parentheses(col) for col in data.columns]

# Apply the formatting to the specific columns
data['Erstellerschlüssel_bezeichner'] = data['Erstellerschlüssel_bezeichner'].apply(lambda x: format_with_leading_zeros(x, 2))
data['Dokumenttyp_bezeichner'] = data['Dokumenttyp_bezeichner'].apply(lambda x: format_with_leading_zeros(x, 3))
data['Leitungsanlage_bezeichner'] = data['Leitungsanlage_bezeichner'].apply(lambda x: format_with_leading_zeros(x, 4))

data = data.applymap(get_pure_dataset)
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")

model = AutoModelForTokenClassification.from_pretrained("Resi/layoutlmv3-multilabel-v2-colab")

dataset = load_dataset("Resi/layoutlmv3-full-annotation-filtered")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

confidence_thresholds = {
    'Dokumenttyp': 99.1,
    'Mast': 80.0, #97.0,
    'Leitungsanlage': 97.0,
    'Ersteller': 98.16
}

column_mapping = {
    'Leitungsanlage': 'Leitungsanlage',
    'Mast': 'Mast',
    'Ersteller': 'Erstellerschlüssel',
    'Dokumenttyp': 'Dokumenttyp'
}

features = dataset["train"].features
column_names = dataset["train"].column_names
image_column_name = "image"
text_column_name = "tokens"
boxes_column_name = "bboxes"
label_column_name = "ner_tags"

if isinstance(features[label_column_name].feature, ClassLabel):
    label_list = features[label_column_name].feature.names
    # No need to convert the labels since they are already ints.
    id2label = {k: v for k,v in enumerate(label_list)}
    label2id = {v: k for k,v in enumerate(label_list)}
else:
    label_list = get_label_list(dataset["train"][label_column_name])
    id2label = {k: v for k,v in enumerate(label_list)}
    label2id = {v: k for k,v in enumerate(label_list)}

num_labels = len(label_list)
def main():
    # from huggingface_hub import notebook_login
    # notebook_login()
    image1 = Image.open(r".\png_with_label_downscaled_1_1_ground_truth\186953-413_D1_Index_03.png")
    image2 = Image.open(r".\png_with_label_downscaled_1_1_ground_truth\319.png")
    image3 = Image.open(r".\png_with_label_downscaled_1_1_ground_truth\000-7510-00000-99-GP-025-00002-00_Lageplan 7510_3_Mast_228_-_Gerüst_EICHS_1_A_13_1.png")
    image4 = Image.open(r".\png_with_label_downscaled_1_1_ground_truth\458.png")
    image5 = Image.open(r".\png_with_label_downscaled_1_1_ground_truth\NBR-1450-078A-01-GA-101-00001-01_Baugrunduntersuchung 1450-078A.png")

    #TODO: Neues giga image als Beispiel einfugen:
    # NBR-1450-078A-01-GA-101-00001-01_Baugrunduntersuchung 1450-078A_page_1
    image1.save("186953-413_D1_Index_03.png")
    image2.save("319.png")
    image3.save("000-7510-00000-99-GP-025-00002-00_Lageplan 7510_3_Mast_228_-_Gerüst_EICHS_1_A_13_1.png")
    image4.save("458.png")
    image5.save("NBR-1450-078A-01-GA-101-00001-01_Baugrunduntersuchung 1450-078A.png")

    #Gradio:
    title = "Document Metadata Extraction: My Bachelor's Thesis Demo"
    description = "Explore my bachelor's thesis project on document metadata extraction! This interactive demo utilizes LayoutLMv3 and Donut to showcase state-of-the-art techniques for understanding and analyzing document images. Upload your own image or try the example to see how the model extracts key information"
    article = "<p style='text-align: center'><a href='https://github.com/Flashness123/Bachelorarbeit/tree/main'>Thesis: Verbesserung der Token-Klassifikation in technischen Zeichnungen durch den Einsatz von Transformer-basierten Modellen</a> | <a href='https://github.com/microsoft/unilm'>Github Repo</a></p>"
    examples =[["186953-413_D1_Index_03.png"], ["319.png"], ["000-7510-00000-99-GP-025-00002-00_Lageplan 7510_3_Mast_228_-_Gerüst_EICHS_1_A_13_1.png"], ["458.png"], ["NBR-1450-078A-01-GA-101-00001-01_Baugrunduntersuchung 1450-078A.png"]]

    css = """.output_image, .input_image {height: 600px !important}"""

    iface = gr.Interface(
        fn=metadata_extraction,
        inputs=gr.Image(type="pil"),
        outputs=gr.HTML(label="Mast Occurrences"),
        #outputs=None,  # No image output needed
        title=title,
        description=description,
        article=article,
        examples=examples,
        css=css
    )
    iface.launch(debug=True, share=True)
if __name__ == "__main__":
    main()





