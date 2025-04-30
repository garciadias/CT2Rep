import json
import re
from collections import Counter

import pandas as pd


class Tokenizer(object):
    def __init__(self, args):
        self.threshold = args.threshold
        self.clean_report = self.clean_report_mimic_cxr
        self.accession_to_text = self.load_accession_text(args.xlsxfile)

        self.token2idx, self.idx2token = self.create_vocabulary()

        with open("idx2token.json", "w") as json_file:
            # Write the dictionary to the file using JSON format
            json.dump(self.idx2token, json_file)

        with open("token2idx.json", "w") as json_file:
            # Write the dictionary to the file using JSON format
            json.dump(self.token2idx, json_file)

    def load_accession_text(self, xlsx_file):
        df = pd.read_excel(xlsx_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row["AccessionNo"]] = row["Findings_EN"]
        return accession_to_text

    def create_vocabulary(self):
        total_tokens = []

        for example in self.accession_to_text.values():
            tokens = self.clean_report(example).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ["<unk>"]
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        # Process the report in two steps
        report_parts = self.clean_report_text_iu_xray(report)
        tokens = [self.clean_sentence_iu_xray(sent) for sent in report_parts if self.clean_sentence_iu_xray(sent) != []]
        report = " . ".join(tokens) + " ."

        return report

    @staticmethod
    def clean_report_text_iu_xray(text):
        # Define all replacements as (pattern, replacement) pairs
        replacements = [
            (r"\n", " "),  # Replace newlines with spaces
            (r"_{2,}", "_"),  # Replace multiple underscores with single underscore
            (r" {2,}", " "),  # Replace multiple spaces with single space
            (r"\.{2,}", "."),  # Replace multiple periods with single period
            (r"^1\. ", ""),  # Remove '1. ' at the start
            (r"\. [2-5]\. ", ". "),  # Replace '. 2. ', '. 3. ', etc. with '. '
            (r" [2-5]\. ", ". "),  # Replace ' 2. ', ' 3. ', etc. with '. '
        ]

        # Apply each replacement in sequence
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)

        # Final processing steps
        return text.strip().lower().split(". ")

    @staticmethod
    def clean_sentence_iu_xray(text):
        # Remove specific quoted characters
        for char in ['"', "/", "\\", "'"]:
            text = text.replace(char, "")

        # Remove punctuation and other special characters
        text = re.sub(r"[.,?;*!%^&_+():-\[\]{}]", "", text)

        return text.strip().lower()

    @staticmethod
    def clean_report_text_mimic_cxr(text):
        # Define all replacements as (pattern, replacement) pairs
        replacements = [
            (r"\n", " "),  # Replace newlines with spaces
            (r"_{2,}", "_"),  # Replace multiple underscores with single underscore
            (r" {2,}", " "),  # Replace multiple spaces with single space
            (r"\.{2,}", "."),  # Replace multiple periods with single period
            (r"^1\. ", ""),  # Remove '1. ' at the start
            (r"\. [2-5]\. ", ". "),  # Replace '. 2. ', '. 3. ', etc. with '. '
            (r" [2-5]\. ", ". "),  # Replace ' 2. ', ' 3. ', etc. with '. '
        ]

        # Apply each replacement in sequence
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)

        # Final processing steps
        return text.strip().lower().split(". ")

    # Clean special characters from sentences
    @staticmethod
    def clean_sentence_mimic_cxr(sentence):
        # Remove specific quoted characters
        for char in ['"', "/", "\\", "'"]:
            sentence = sentence.replace(char, "")

        # Remove punctuation and other special characters
        sentence = re.sub(r"[.,?;*!%^&_+():-\[\]{}]", "", sentence)

        return sentence.strip().lower()

    def clean_report_mimic_cxr(self, report):
        # Process the report in two steps
        report_parts = self.clean_report_text_mimic_cxr(report)
        tokens = [
            self.clean_sentence_mimic_cxr(sent) for sent in report_parts if self.clean_sentence_mimic_cxr(sent) != []
        ]
        report = " . ".join(tokens) + " ."

        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx["<unk>"]
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ""
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += " "
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
