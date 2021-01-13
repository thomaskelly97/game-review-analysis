import json_lines
from googletrans import Translator, constants
from google_trans_new import google_translator
from pprint import pprint
import unidecode
import codecs
from transliterate import translit
import ftfy
import json
from pprint import pprint

# # NORMAL LANGUAGES
translator = google_translator()


def translate_cyrillic(text):
    decoded = ftfy.fix_text(text)
    return decoded


write_file = open("../data/new_data.jl", "w")

with codecs.open("../data/reviews_37.jl", "rb") as f:
    for item in json_lines.reader(f):
        print("=======================================")

        text = item['text']
        print("Raw text: ", text)
        text = translate_cyrillic(text)

        translation = translator.translate(text)
        text = translation
        print("Translated Text: ", text)
        build_obj = {
            "text": text,
            "voted_up": item['voted_up'],
            "early_access": item['early_access']
        }
        write_file.write(json.dumps(build_obj))
