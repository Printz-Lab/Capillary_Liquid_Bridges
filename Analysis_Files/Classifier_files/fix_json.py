

raw = open(json_path).read()
# insert missing commas between adjacent objects
raw = raw.replace("}{", "},{")
# remove any trailing commas before the final ]
raw = raw.replace(",]", "]")
masks = json.loads(raw)
