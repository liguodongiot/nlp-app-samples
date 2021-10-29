

import logging
import os
from typing import List, Optional, Union, Dict
from torch.utils.data.dataset import Dataset, IterableDataset
from transformers import PreTrainedTokenizer
from transformers.data.processors.glue import glue_convert_examples_to_features
from torch import nn

from tests.iter_version_dev.V1_0_0.common import Split,TaskName
from tests.iter_version_dev.V1_0_0.processor_bert_ner import InputExample,InputFeatures,TokenClassProcessor

logger = logging.getLogger(__name__)

def convert_texts_to_features(
    texts,
    tokenizer: PreTrainedTokenizer,
    max_seq_length=512,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:

    features = []
    for (ex_index, example) in enumerate(texts):
        tokens = []
        word_tokens = tokenizer.tokenize(example)
        # bert-base-multilingual-cased sometimes output "nothing ([]) when
        # calling tokenize with just a space.
        if len(word_tokens) > 0:
            tokens.extend(word_tokens)

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1]
                          * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] *
                           padding_length) + segment_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                label_ids=None
            )
        )
    return features


def convert_examples_to_features(
    examples: List[InputExample],
    label_map,
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """ 
        Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # clean up all this to leverage built-in features of tokenizers

    # label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when
            # calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and
                # padding ids for the remaining tokens
                if label not in label_map:
                    label = 'O'
                    logger.error('数据集出现未知标签' + label + '，强制转换为 O ')
                label_ids.extend([label_map[label]] +
                                 [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1]
                          * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] *
                           padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join(
                [str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join(
                [str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids
            )
        )
    return features

class NerDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    features: List[InputFeatures]
    pad_token_label_id = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        label_map,
        pretrain_type: str,
        max_seq_length: Optional[int] = 128,
        mode: Split = Split.train,
        task_name="ner"
    ):
        processor = TokenClassProcessor()

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        logger.info(f"Creating features from dataset file at {data_dir}")
        examples = processor.get_examples(data_dir, mode)
        # clean up all this to leverage built-in features of
        # tokenizers
        self.features = convert_examples_to_features(
            examples,
            label_map,
            max_seq_length,
            tokenizer,
            cls_token_at_end=bool(pretrain_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if pretrain_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
          
            pad_on_left=bool(tokenizer.padding_side == "left"),
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


class NerIterableDataset(IterableDataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """
    features: List[InputFeatures]
    pad_token_label_id = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        label_map,
        pretrain_type: str,
        max_seq_length: Optional[int] = 128,
        mode: Split = Split.train,
        task_name="ner"
    ):
        processor = TokenClassProcessor()

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        logger.info(f"Creating features from dataset file at {data_dir}")
        examples = processor.get_examples(data_dir, mode)
        # clean up all this to leverage built-in features of
        # tokenizers
        self.features = convert_examples_to_features(
            examples,
            label_map,
            max_seq_length,
            tokenizer,
            cls_token_at_end=bool(pretrain_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if pretrain_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            # sep_token_extra=bool(pretrain_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences,
            # cf.
            # github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(tokenizer.padding_side == "left"),
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id,
        )

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        for feature in self.features:
            yield feature


class NerPredictDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    features: List[InputFeatures]
    pad_token_label_id = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        texts,
        id2label,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: Optional[int] = 512,
        mode: Split = Split.train,
        task_name="ner"
    ):
        self.features = convert_texts_to_features(
            texts,
            tokenizer,
            max_seq_length=max_seq_length
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

