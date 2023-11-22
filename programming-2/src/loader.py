'''
@author: jcheung

Developed for Python 2. Automatically converted to Python 3; may result in bugs.
'''
import xml.etree.cElementTree as ET

class WSDInstance:
    def __init__(self, my_id, lemma, context, index):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index      # index of lemma within the context
    def __str__(self):
        '''
        For printing purposes.
        '''
        return '{}\t{}\t{}\t{}'.format(self.id, self.lemma, ' '.join(self.context), self.index)

def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()
    
    dev_instances = {}
    test_instances = {}
    
    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            # construct sentence context
            context = [el.attrib['lemma'] for el in sentence]
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = el.attrib['lemma']
                    instances[my_id] = WSDInstance(my_id, lemma, context, i)
    return dev_instances, test_instances

def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys. 
    '''
    dev_key = {}
    test_key = {}
    with open(f) as file:
        for line in file:
            if len(line) <= 1: continue
            doc, my_id, sense_key = line.strip().split(' ', 2)
            key = dev_key if doc == 'd001' else test_key
            key[my_id] = sense_key.split()
    return dev_key, test_key

if __name__ == '__main__':
    data_f = '../data/multilingual-all-words.en.xml'
    key_f = '../data/wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)
    
    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for k, v in dev_instances.items() if k in dev_key}
    test_instances = {k: v for k, v in test_instances.items() if k in test_key}
    
    print(len(dev_instances)) # number of dev instances
    print(len(test_instances)) # number of test instances

    # print number of distinct lemmas
    print(len(set(instance.lemma for instance in dev_instances.values())))
    print(len(set(instance.lemma for instance in test_instances.values())))

    # print overlap of lemmas between dev and test
    print(len(set(instance.lemma for instance in dev_instances.values()) & set(instance.lemma for instance in test_instances.values())))
    