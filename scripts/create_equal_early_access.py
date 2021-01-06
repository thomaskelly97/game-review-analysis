import json_lines
import json

X = []
y = []
z = []

write_file = open("../data/balanced_early_access1.jl", "w")

countnegs = 0
with open("../data/new_data.jl", "rb") as f:
    for item in json_lines.reader(f):
        text = item['text']
        vote = item["voted_up"]
        early = item["early_access"]

        if early == True:
            countnegs = countnegs + 1
            print("NEGATIVE COUNT--> ", countnegs)
            if countnegs < 501:
                build_obj = {
                    "text": text,
                    "voted_up": item['voted_up'],
                    "early_access": item['early_access']
                }
                write_file.write(json.dumps(build_obj))
