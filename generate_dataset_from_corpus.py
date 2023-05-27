#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tqdm
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import tensorflow.compat.v1 as tf

import model, sample, encoder

def load_dataset(input_path, encoding=None):
    paths = []
    if os.path.isfile(input_path):
        # Simple file
        paths.append(input_path)
    elif os.path.isdir(input_path):
        # Directory
        for (dirpath, _, fnames) in os.walk(input_path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(input_path)

    records = []
    empty_count = 0

    for path in paths:
        if path.endswith('.txt'):
            # Plain text
            l_title = ''
            l_content = ''
            with open(path, 'r', encoding=encoding) as fp:
                for line in fp:
                    l = line.rstrip('\n').strip()

                    if '<|endoftext|>' in l:
                        if l_title == '':
                            empty_count = empty_count + 1
                        records.append([l_title, l_content, 'Real'])
                        l_title = ''
                        l_content = ''
                    elif l.endswith('@@@'):
                        l_title = l.replace('@@@', '').strip()
                        l_content = ''
                    else:
                        l_content = l_content + line

    df = pd.DataFrame.from_records(records, columns=['topic', 'content', 'label'])
    print("=" * 20 + " Read from " + input_path + ", number of records: " + str(len(df)) + " , empty title: " + str(empty_count))
    return df


def clean_text(input_text):
    text = input_text
    if '<|endoftext|>' in text:
        end = text.find('<|endoftext|>')
        text = text[0:end]
    if '@@@' in text:
        start = text.find('@@@') + 3
        if start < len(text):
            text = text[start:]
    text = text.strip()
    return text


def generate_from_dataset(
    model_name='355M-v1',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
    encoding='utf-8',
    input_dataset='./datasets/CASIE_finetune.txt',
    output_file='./output/output.xlsx'
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)

    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        input_df = load_dataset(input_dataset, encoding)

        outputs = []
        for ind, row in tqdm.tqdm(input_df.iterrows()):
            raw_text = row['topic']
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    text = clean_text(text)
                    outputs.append([raw_text, text, 'Fake'])
                    #print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    #print(text)
            #print("=" * 80)

        output_df = pd.DataFrame.from_records(outputs, columns=['topic', 'content', 'label'])
        wb = Workbook()
        ws = wb.active
        for row in dataframe_to_rows(output_df, index=False, header=True):
            ws.append(row)
        for row in dataframe_to_rows(input_df, index=False, header=False):
            ws.append(row)
        wb.save(output_file)
        print("=" * 20 + " " + str(len(output_df)) + " lines of output has written successfully to: " + output_file)

if __name__ == '__main__':
    fire.Fire(generate_from_dataset)