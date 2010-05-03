resource_path = 'training/'
test_prefix = ('test', )
training_prefix = ('train', )

line_delimiters = ('*', '\n', '=')
word_delimiters = ('[', ']')
sentence_separator = ('.', '?', '!')

START = '<s>'
END = '</s>'

def normalize_word(word):
    return word.strip().upper()

def filter_unused_lines(line):
    """ extracts all relevant lines """
    return not line.startswith(line_delimiters)

def filter_unused_strings(string):
    """ extracts all word/tag constructs """
    return not (string in word_delimiters)

def filter_start_end_states(state):
    return not (state in (START, END))

def map_extract_word(term):
    return term.split('/')[0]

def map_extract_tag(term):
    return term.split('/')[1].split('|')[0]

def parse(file):
    """ parses the file in the 'Penn Treebank annotation style for POS tags' into a list of sequences of word/tag constructs """
    line_list = file.readlines()
    line_list = filter(filter_unused_lines, line_list)
    sentence_list = []
    sentence = []
    for line in line_list:
        line = filter(filter_unused_strings, line)
        term_list = line.split()
        for term in term_list:
            sentence.append(term)
            if term.startswith(sentence_separator):
                s = [tuple(x.split('/')) for x in sentence]
                s = [(normalize_word(x[0]), x[1]) for x in s]
                sentence_list.append(s)
                sentence = []
    return sentence_list