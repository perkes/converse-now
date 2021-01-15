# Named Entity Extractor

A tool to extract specific named entities from text fragments.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- pip
- virtualenv
- A valid configuration file (see below).
- Missing files in the data folder: https://drive.google.com/drive/folders/1rcAz20u9rh2vW7OS8ORtgwcKRa81ys0g?usp=sharing

To get **pip**: download [get-pip.py](https://bootstrap.pypa.io/get-pip.py).
```
python get-pip.py
```
Pip is now installed!

To get **virutalenv**: 
```
pip install virtualenv
``` 

### Installing

Create a new virtual environment:
```
virtualenv -p python3 venv
```    

Start the virtual environment:
```
source venv/bin/activate
```    
Install requirements with pip:
```
pip install -r requirements.txt
```
Download the contents of data.zip into the data folder.

### Configuring

#### Configuring the CSV to JSON configuration file

- **epochs**: the number of epochs to train the model for.
- **dropout_rate**: the droput rate for the model.
- **revisions_file**: the route to the revisions file for the model (if we train using only the new entities, that could have detrimental effects for model performance related to previously existing entities).
- **train_test_split**: the portion of the dataset used for training (as opposed to testing), a number between 0 and 1. Currently not implemented.
- **model_name**: a name for the model, any string will do.
- **model_output_dir**: the output directory for the trained model, also used as the input directory during testing.
- **test_input**: the route to the validation dataset used during testing.
- **test_output**: the route to the output during the testing phase. The content of the output file will be the same as the input file, plus the model's predictions.
- **review_column**: the name of the column containing the review text in "test_input".
- **entities**: a list of dictionaries, with the keys following below.
    - **label**: the NER label of the new entity. 
    - **sources**: the files used as sources for terms associated to this entity.
    - **source_column**: the name of the column containing the aforementioned text in the source files.
    - **templates**: route to the file containing sentence templates for this entity.
    - **max_length**: maximum number of words for a given example.
    - **filter_characters**: characters to be filtered, represented as a string.
    - **keep_before**: characters after which nothing else is read. For instance if ',' is included 'Pears, 1 portion' would be transformed into 'Pears'.
    - **filter_words**: instances containing words in this list will be excluded.
    - **multiplier**: a natural number. For instance if this is 2, every instance will be included twice. Used to ameliorate class imbalances.
 
### Training

To run the training script:
```
python train.py
```

### Testing

To run the testing script:
```
python test.py
```

## Built With

* [Pandas](https://pandas.pydata.org/) - Data manipulation.
* [spaCy](https://spacy.io//) - Industrial-Strength Natural Language Processing.

## Authors

* **Jonathan Perkes**
