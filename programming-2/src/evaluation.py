def accuracy(predictions, gold_keys):
    """Calculate the accuracy of WSD predictions."""
    correct = 0
    for instance_id, predicted_synset in predictions.items():
        correct_senses = gold_keys.get(instance_id, [])
        # get the keys of all lemmas in the predicted synset
        predicted_keys = [lemma.key() for lemma in predicted_synset.lemmas()] if predicted_synset else []
        # check if any of the predicted keys match the correct senses
        if any(predicted_key in correct_senses for predicted_key in predicted_keys):
            correct += 1
    return correct / len(predictions) if predictions else 0

def get_misclassified(predictions, gold_keys):
    """Return a list of misclassified instances."""
    misclassified = []
    for instance_id, predicted_synset in predictions.items():
        correct_senses = gold_keys.get(instance_id, [])
        predicted_keys = [lemma.key() for lemma in predicted_synset.lemmas()] if predicted_synset else []
        if not any(predicted_key in correct_senses for predicted_key in predicted_keys):
            misclassified.append(instance_id)
    return misclassified

def print_evaluation(test_instances, predictions, gold_keys):
    print('Accuracy: ', accuracy(predictions, gold_keys))
    misclassified = get_misclassified(predictions, gold_keys)
    print('Misclassified: ', len(misclassified))
    for instance_id in misclassified[:10]:
        print('Misclassified instance: ', instance_id)
        print('     Lemma: ', test_instances[instance_id].lemma)
        print('     Context: ', ' '.join(test_instances[instance_id].context))
        print('     Prediction: ', [lemma.key() for lemma in predictions[instance_id].lemmas()])
        print('     Gold: ', gold_keys[instance_id])
