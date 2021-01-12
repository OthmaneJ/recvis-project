# coding: utf-8
"""
Data module
"""
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch
import numpy as np
# import sys
# sys.path.append('//content//drive//MyDrive///MVA//reconaissance objet//projet//recvis-project//slt')

def load_dataset_file(filename):
    print(filename)
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def load_dope_dataset(filename):
  with open(filename, 'rb') as file_:
      loaded_object = pickle.load(file_)
      return loaded_object

class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        path_dope: str,
        kind_data: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            path_dope: Common prefix for paths to the dope dataset.
            kind_data: Data considered dev or train or test
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("body_dope", fields[3]),
                ("face_dope", fields[4]),
                ("gls", fields[5]),
                ("txt", fields[6]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        print("The path is: {}".format(path))
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for key in tmp:
                s = tmp[key]
                num_frames = s['sign'].size(0)
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )

                else:

                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                        "body_dope": torch.from_numpy(0*s['body_2d'].reshape((num_frames, -1)).astype('float32')),
                        "face_dope": torch.from_numpy(s['face_2d'].reshape((num_frames, -1)).astype('float32'))
                    }
        """print('dope path {}'.format(path_dope))
        tmp = load_dope_dataset(path_dope)
        for key in tmp.keys():
            seq_id = kind_data + '/' + key.strip('.mp4')
            try:
                assert seq_id in samples, "the sequence {} is not in keys".format(seq_id)
            except:
                continue

            samples[seq_id]['body_dope'], samples[seq_id]['face_dope'] = [], []
            for i in range(len(tmp[key])):
                if len(tmp[key][i]['body']) > 0:
                    samples[seq_id]['body_dope'].append(tmp[key][i]['body'][0]['pose3d'].reshape(-1)) 
                else: 
                    samples[seq_id]['body_dope'].append(np.zeros(39)) 
                if len(tmp[key][i]['face']) > 0:
                    samples[seq_id]['face_dope'].append(tmp[key][i]['face'][0]['pose3d'].reshape(-1))
                else:
                    samples[seq_id]['face_dope'].append(np.zeros(252))
            samples[seq_id]['body_dope'] = torch.from_numpy(np.vstack(samples[seq_id]['body_dope']))
            samples[seq_id]['face_dope'] = torch.from_numpy(np.vstack(samples[seq_id]['face_dope']))               
        """
        examples = []
        for s in samples:
            sample = samples[s]
            # if 'body_dope' not in samples:
            #     sample['body_dope'] = torch.zeros((sample['sign'].size(0), 39))
            #     sample['face_dope'] = torch.zeros((sample['sign'].size(0), 252))
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample['body_dope'],
                        sample['face_dope'],
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
