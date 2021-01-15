import spacy
import json
import csv

def find_occasion(review):
    if review.find('dinner') >= 0:
        return 'dinner'
    elif review.find('lunch') >= 0:
        return 'lunch'
    elif review.find('breakfast') >= 0:
        return 'breakfast'
    elif review.find('brunch') >= 0:
        return 'brunch'
    return ''

with open('config.json', encoding='utf-8') as config_file:
    config = json.load(config_file)

nlp = spacy.load(config['model_output_dir'])

reader = csv.DictReader(open(config['test_input'], 'r'))
writer = csv.DictWriter(open(config['test_output'], 'w'), fieldnames = reader.fieldnames + [field + '_pred' for field in reader.fieldnames if field != config['review_column']])
writer.writeheader()

for row in reader:
    text = row[config['review_column']]
    doc = nlp(row[config['review_column']])
    row['occasion_pred'] = find_occasion(text)
    keys = [e['label'].lower() for e in config['entities']]
    for ent in doc.ents:
        if ent.label_.lower() in keys:
            key = ent.label_.lower() + '_pred'
            row[key] = [] if not key in row else row[key]
            row[key].append(ent.text)
    writer.writerow(row)