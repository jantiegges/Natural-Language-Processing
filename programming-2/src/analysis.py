import pandas as pd
from loader import load_key, load_instances
from evaluation import accuracy, get_misclassified, print_evaluation
from nltk.corpus import wordnet as wn

if __name__ == '__main__':
    data_f = '../data/multilingual-all-words.en.xml'
    key_f = '../data/wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)
    test_instances = {k: v for k, v in test_instances.items() if k in test_key}

    # load predictions from out folder
    with open('../out/baseline_predictions.txt') as f:
        mf_pred = f.read().splitlines()
    with open('../out/lesk_predictions.txt') as f:
        lesk_pred = f.read().splitlines()
    with open('../out/bert_predictions.txt') as f:
        bert_pred = f.read().splitlines()
    with open('../out/nb_predictions.txt') as f:
        nb_pred = f.read().splitlines()


    mf_pred = [row.split(' ') for row in mf_pred]
    lesk_pred = [row.split(' ') for row in lesk_pred]
    bert_pred = [row.split(' ') for row in bert_pred]
    nb_pred = [row.split(' ') for row in nb_pred]
    mf_pred_df = pd.DataFrame(mf_pred, columns=['instance_id', 'prediction'])
    lesk_pred_df = pd.DataFrame(lesk_pred, columns=['instance_id', 'prediction'])
    bert_pred_df = pd.DataFrame(bert_pred, columns=['instance_id', 'prediction'])
    nb_pred_df = pd.DataFrame(nb_pred, columns=['instance_id', 'prediction'])

    # convert synset name to synset object
    mf_pred_df['prediction'] = mf_pred_df['prediction'].apply(lambda x: x[8:-2] if x != 'None' else None)
    mf_pred_df['prediction'] = mf_pred_df['prediction'].apply(lambda x: wn.synset(x) if x != None else None)
    mf_pred_dict = mf_pred_df.set_index('instance_id').to_dict()['prediction']

    lesk_pred_df['prediction'] = lesk_pred_df['prediction'].apply(lambda x: x[8:-2] if x != 'None' else None)
    lesk_pred_df['prediction'] = lesk_pred_df['prediction'].apply(lambda x: wn.synset(x) if x != None else None)
    lesk_pred_dict = lesk_pred_df.set_index('instance_id').to_dict()['prediction']

    bert_pred_df['prediction'] = bert_pred_df['prediction'].apply(lambda x: x[8:-2] if x != 'None' else None)
    bert_pred_df['prediction'] = bert_pred_df['prediction'].apply(lambda x: wn.synset(x) if x != None else None)
    bert_pred_dict = bert_pred_df.set_index('instance_id').to_dict()['prediction']

    nb_pred_df['prediction'] = nb_pred_df['prediction'].apply(lambda x: x[8:-2] if x != 'None' else None)
    nb_pred_df['prediction'] = nb_pred_df['prediction'].apply(lambda x: wn.synset(x) if x != None else None)
    nb_pred_dict = nb_pred_df.set_index('instance_id').to_dict()['prediction']

    # print misclassified instances
    print_evaluation(test_instances, mf_pred_dict, test_key)
    print_evaluation(test_instances, lesk_pred_dict, test_key)
    print_evaluation(test_instances, bert_pred_dict, test_key)
    print_evaluation(test_instances, nb_pred_dict, test_key)