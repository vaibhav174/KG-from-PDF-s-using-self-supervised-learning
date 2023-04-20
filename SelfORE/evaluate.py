import yaml

from SelfORE.model.classifier import Classifier


def eval_class(sentence_path=None, model_path=None):
    # read config file and create variables
    config = yaml.load(open("SelfORE/model/config.yaml", "r"), Loader=yaml.FullLoader)
    k = config['k']
    sent_path = config['sentence_path'] #config['test']['sentence_path']
    max_len = config['bert_max_len']
    b_size = config['batch_size']
    epoch = config['epoch']

    if model_path is None:
        model_path = config['model_path']

    # load model
    classifier = Classifier(k, sent_path, max_len, b_size, epoch)

    max_idx = classifier.inference(model_path=model_path)

    return max_idx


def eval_class_supervised():
    # read config file aand create variables
    config = yaml.load(open("model/config_trex.yaml", "r"), Loader=yaml.FullLoader)
    k = config['k']
    sent_path = config['sentence_path'] #config['test']['sentence_path']
    max_len = config['bert_max_len']
    b_size = config['batch_size']
    epoch = config['epoch']

    # load model
    classifier = Classifier(k, sent_path, max_len, b_size, epoch, rel_available=True)
    classifier.test(config['model_path'])


if __name__ == "__main__":
    eval_class_supervised()
