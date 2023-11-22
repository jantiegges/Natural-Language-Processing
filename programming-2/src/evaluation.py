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
