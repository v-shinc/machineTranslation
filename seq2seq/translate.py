import os
import sys
import json
import tensorflow as tf
from data_helper import DataSet
from model import Model
def translate(dir_path):
    checkpoint_dir = os.path.abspath(os.path.join(dir_path, "checkpoints"))
    save_path = os.path.join(checkpoint_dir, "model")
    config_path = os.path.join(dir_path, 'config.json')
    params = json.load(open(config_path))
    params['reload'] = True
    params['load_path'] = save_path
    with tf.Graph().as_default():
        model = Model(params)
    dataset = DataSet(params['dictionaries'], params['max_len'], params['source_vocab_size'], params['target_vocab_size'])

    sentences = ['Access to the Internet is itself a fundamental right .',
                'These schools then subsidised free education for the non - working poor .',
                 'What will they do ? CSSD lacks knowledge of both Voldemort and candy bars in Prague .']
    xs, x_mask = dataset.create_model_input(sentences)

    for x, mask in zip(xs, x_mask):
        print x, mask
        print dataset.to_readable_source(x)
        ys, scores = model.predict(x[None, :], mask[None, :], 10, params['max_len'])
        for i, y in enumerate(ys):
            print dataset.to_readable(y), scores[i]

if __name__ == '__main__':
    translate(sys.argv[1])
