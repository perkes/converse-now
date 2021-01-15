from spacy.util import minibatch, compounding
from tqdm import tqdm

import pandas as pd
import warnings
import random
import string
import spacy
import json
import csv
import re

def generate_dataset(terms, label, templates):
    dataset = []
    pbar = tqdm(total = len(terms))
    while len(terms) >= 3:
        entities = []
        template = templates[random.randint(0, len(templates) - 1)]
        # find out how many braces "{}" need to be replaced in the template
        matches = re.findall('{}', template)
        # for each brace, replace with a food entity from the shuffled food data
        for match in matches:
            term = terms.pop()
            pbar.update(1)
            # replace the pattern, but then find the match of the food entity we just inserted
            template = template.replace(match, term, 1)
            match_span = re.search(term, template).span()
            # use that match to find the index positions of the food entity in the sentence, append
            entities.append((match_span[0], match_span[1], label))
            
        overlapping = [[x,y] for x in entities for y in entities if x is not y and x[1] >= y[0] and x[0] <= y[0] and x is not y]
        if not overlapping:
            dataset.append((template, {'entities': entities}))
    pbar.close()
    return dataset

def generate_revisions(nlp, onfig):
    npr_df = pd.read_csv(config['revisions_file'])
    revision_texts = []
    
    print('Converting revision articles to spacy objects...')
    docs = nlp.pipe(npr_df['Article'][:6000], batch_size = 30, disable = ['tagger', 'ner'])
    pbar = tqdm(total = 6000)
    for doc in docs:
        pbar.update(1)
        for sentence in doc.sents:
            if  40 < len(sentence.text) < 80:
                # some of the sentences had excessive whitespace in between words, so we're trimming that
                revision_texts.append(' '.join(re.split('\s+', sentence.text, flags = re.UNICODE)))
    pbar.close()
    revisions = []
    print('Converting revision articles to spacy dataset format...')
    docs = nlp.pipe(revision_texts, batch_size = 50, disable=['tagger', 'parser'])
    pbar = tqdm(total = len(revision_texts))
    for doc in docs:
        pbar.update(1)
        # don't append sentences that have no entities
        if len(doc.ents) > 0:
            revisions.append((doc.text, {'entities': [(e.start_char, e.end_char, e.label_) for e in doc.ents]}))
    pbar.close()
    return revisions

def get_values(entity):
    values = set()
    for file in entity['sources']:
        reader = csv.DictReader(open(file, 'r'))
        for row in reader:
            values.add(row[entity['source_column']].lower())
            
    for c in entity['keep_before']:
        values = [v.split(c)[0] for v in values]
    
    values = [''.join([c for c in v if valid_char(c, entity)]).strip().lower() for v in values]
    values = set([''.join([c for c in v if valid_char(c, entity)]) for v in values if valid_value(v, entity)])
    
    return list(values) * entity['multiplier']

def valid_char(c, entity):
    is_digit = c in string.digits
    is_excluded = c in entity['filter_characters']
    return not is_digit and not is_excluded

def valid_value(value, entity):
    return len(value.split()) <= entity['max_length'] and all([value.find(w) == -1 for w in entity['filter_words']])

def train_test_model(config):
    nlp = spacy.load("en_core_web_lg")
    ner = nlp.get_pipe('ner')
    TRAIN_DATA, TEST_DATA = [], []
    
    for entity in config['entities']:
        f = open(entity['templates'], 'r')
        templates = f.readlines()
    
        values = get_values(entity)
        print('Generating dataset for label ' + entity['label'] + '...')
        dataset = generate_dataset(values, entity['label'], templates)

        random.shuffle(dataset)
        pivot = int(len(dataset) * config['train_test_split'])
        TRAIN, TEST = dataset[:pivot], dataset[pivot:]
        
        TRAIN_DATA += TRAIN
        TEST_DATA += TEST
        
        ner.add_label(entity['label'])

    revisions = generate_revisions(nlp, config)           
    random.shuffle(revisions)
    pivot = int(len(revisions) * config['train_test_split'])
    TRAIN_REVISIONS, TEST_REVISIONS = revisions[:pivot], revisions[pivot:]

    TRAIN_DATA = TRAIN_REVISIONS + TRAIN_DATA
    TEST_DATA = TEST_REVISIONS + TEST_DATA

    # get the names of the components we want to disable during training
    pipe_exceptions = ['ner', 'trf_wordpiecer', 'trf_tok2vec']
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    epochs = config['epochs']
    optimizer = nlp.resume_training()
    print('Training model...')
    with nlp.disable_pipes(*other_pipes):
        sizes = compounding(1.0, 4.0, 1.001)
        
        for epoch in range(epochs):
            examples = TRAIN_DATA
            random.shuffle(examples)
            batches = minibatch(examples, size = sizes)
            losses = {}

            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd = optimizer, drop = 0.35, losses = losses)

            print("Losses ({}/{})".format(epoch + 1, epochs), losses)

    nlp.meta["name"] = config['model_name']
    nlp.to_disk(config['model_output_dir'])
    
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    with open('config.json', encoding='utf-8') as config_file:
        config = json.load(config_file)
        train_test_model(config)