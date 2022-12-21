import spacy
import subprocess
from collections import Counter
from nltk.stem.porter import *
from nltk.tokenize import wordpunct_tokenize
from transformers import BertTokenizer

KP_SEP = ';'
TITLE_SEP = '[sep]'
PRESENT_EOS = '[psep]'
DIGIT = '[digit]'

stemmer = PorterStemmer()

SPECIAL_TOKENS = [KP_SEP, TITLE_SEP, PRESENT_EOS]
UNUSED_TOKEN_MAP = {
    '[unused0]': PRESENT_EOS,
    '[unused1]': KP_SEP
}


class SpacyTokenizer(object):

    def __init__(self, **kwargs):
        model = kwargs.get('model', 'en')
        nlp_kwargs = {'parser': False, 'tagger': False, 'entity': False}
        self.nlp = spacy.load(model, **nlp_kwargs)

    def tokenize(self, text):
        doc = self.nlp(text)
        return [token.text for token in doc]

    @property
    def vocab(self):
        return None


class WhiteSpaceTokenizer(object):

    def __init__(self):
        pass

    def tokenize(self, text):
        return text.split()

    @property
    def vocab(self):
        return None


class MultiprocessingTokenizer(object):

    def __init__(self, args):
        self.args = args
        if self.args['tokenizer'] == 'BertTokenizer':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.args['tokenizer'] == 'SpacyTokenizer':
            self.tokenizer = SpacyTokenizer(model='en_core_web_sm')
        elif self.args['tokenizer'] == 'WhiteSpace':
            self.tokenizer = WhiteSpaceTokenizer()
        else:
            raise ValueError('Unknown tokenizer type!')

    def initializer(self):
        pass

    @property
    def vocab(self):
        return self.tokenizer.vocab

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        return ' '.join(tokens)

    def process(self, example):
        title = example['title'].strip().lower()
        abstract = example['abstract'].strip().lower()

        if self.args['kp_separator']:
            keywords = example['keyword'].lower().split(self.args['kp_separator'])
            keywords = [kp.strip() for kp in keywords]
            present_keywords, absent_keywords = separate_present_absent(
                title + ' ' + abstract, keywords
            )
        else:
            present_keywords = example['present_keywords']
            absent_keywords = example['absent_keywords']

        # filtering empty keyphrases
        present_keywords = [pkp for pkp in present_keywords if pkp]
        absent_keywords = [akp for akp in absent_keywords if akp]

        if self.args['replace_digit_tokenizer']:
            title = fn_replace_digits(title, tokenizer=self.args['replace_digit_tokenizer'])
            abstract = fn_replace_digits(abstract, tokenizer=self.args['replace_digit_tokenizer'])
            present_keywords = [fn_replace_digits(pkp, tokenizer=self.args['replace_digit_tokenizer'])
                                for pkp in present_keywords]
            absent_keywords = [fn_replace_digits(akp, tokenizer=self.args['replace_digit_tokenizer'])
                               for akp in absent_keywords]

        title_tokenized = self.tokenize(title)
        abstract_tokenized = self.tokenize(abstract)
        pkp_tokenized = [self.tokenize(pkp) for pkp in present_keywords]
        akp_tokenized = [self.tokenize(akp) for akp in absent_keywords]

        # TODO: how can we add [digit] as never_split token in BertTokenizer?
        title_tokenized = title_tokenized.replace('[ digit ]', DIGIT)
        abstract_tokenized = abstract_tokenized.replace('[ digit ]', DIGIT)
        pkp_tokenized = [pkp.replace('[ digit ]', DIGIT) for pkp in pkp_tokenized]
        akp_tokenized = [akp.replace('[ digit ]', DIGIT) for akp in akp_tokenized]

        return {
            'id': example['id'],
            'title': {
                'text': title,
                'tokenized': title_tokenized
            },
            'abstract': {
                'text': abstract,
                'tokenized': abstract_tokenized
            },
            'present_kps': {
                'text': present_keywords,
                'tokenized': pkp_tokenized
            },
            'absent_kps': {
                'text': absent_keywords,
                'tokenized': akp_tokenized
            }
        }

    def process_summarization(self, example):
        article = example['article'].strip().lower()
        abstract = example['abstract'].strip().lower()
        
        if self.args['replace_digit_tokenizer']:
            article = fn_replace_digits(article, tokenizer=self.args['replace_digit_tokenizer'])
            abstract = fn_replace_digits(abstract, tokenizer=self.args['replace_digit_tokenizer'])
            
        article_tokenized = self.tokenize(article)
        abstract_tokenized = self.tokenize(abstract)

        # TODO: how can we add [digit] as never_split token in BertTokenizer?
        article_tokenized = article_tokenized.replace('[ digit ]', DIGIT)
        abstract_tokenized = abstract_tokenized.replace('[ digit ]', DIGIT)
        
        return {
            'id': example['id'],
            'article': {
                'text': article,
                'tokenized': article_tokenized
            },
            'abstract': {
                'text': abstract,
                'tokenized': abstract_tokenized
            }
        }


def create_vocab(dataset):
    vocabulary = Counter()
    for ex in dataset:
        vocabulary.update(ex['title']['tokenized'].split())
        vocabulary.update(ex['abstract']['tokenized'].split())
        pkp_tokens = [kp.split() for kp in ex['present_kps']['tokenized']]
        akp_tokens = [kp.split() for kp in ex['absent_kps']['tokenized']]
        kp_tokens = [token for kp in pkp_tokens + akp_tokens for token in kp]
        vocabulary.update(kp_tokens)

    vocab_items = list()
    for (token, freq) in vocabulary.most_common():
        vocab_items.append(token)
    # prepend to list
    for token in SPECIAL_TOKENS:
        if token not in vocab_items:
            vocab_items.insert(0, token)

    return vocab_items


def create_vocab_summarization(dataset):
    vocabulary = Counter()
    for ex in dataset:
        vocabulary.update(ex['article']['tokenized'].split())
        vocabulary.update(ex['abstract']['tokenized'].split())

    vocab_items = list()
    for (token, freq) in vocabulary.most_common():
        vocab_items.append(token)
    # prepend to list
    for token in SPECIAL_TOKENS:
        if token not in vocab_items:
            vocab_items.insert(0, token)

    return vocab_items


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').strip().split(' ')
    return int(num[0])


def stem_word_list(word_list):
    return [stemmer.stem(w.strip().lower()) for w in word_list]


def stem_text(text):
    return ' '.join(stem_word_list(text.split()))


def fn_replace_digits(text, tokenizer='wordpunct'):
    out_tokens = []
    in_tokens = wordpunct_tokenize(text) \
        if tokenizer == 'wordpunct' else text.split()
    for tok in in_tokens:
        if re.match('^\d+$', tok):
            out_tokens.append(DIGIT)
        else:
            out_tokens.append(tok)
    return ' '.join(out_tokens)


def separate_present_absent(source_text, keyphrases):
    present_kps = []
    absent_kps = []
    stemmed_source = stem_text(source_text)
    for kp in keyphrases:
        stemmed_kp = stem_text(kp)
        if stemmed_kp in stemmed_source:
            present_kps.append(kp)
        else:
            absent_kps.append(kp)

    return present_kps, absent_kps
