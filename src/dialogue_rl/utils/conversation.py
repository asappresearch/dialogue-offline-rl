from typing import List, Dict
import pandas as pd

from dialogue_offline_rl.utils.constants import (
    CONVO_START,
    CONVO_END,
    CUS_START,
    CUS_END,
    REP_START,
    REP_END
)


def convo_to_text_abcd(
    utterance_rows: List[List[str]],
    include_start: bool = True,
    separator: str = "",
    convo_start: str = CONVO_START,
    convo_end: str = CONVO_END,
    cus_start: str = CUS_START,
    cus_end: str = CUS_END,
    rep_start: str = REP_START,
    rep_end: str = REP_END,
):
    parts: List[str] = []
    if include_start:
        parts.append(convo_start)

    for idx, row in enumerate(utterance_rows):
        sender_type, utterance = row
        if (sender_type == 'customer'):
            start, end = cus_start, cus_end
        elif (sender_type == 'agent'):
            start, end = rep_start, rep_end
        elif (sender_type == 'action'):
            continue
        else:
            raise ValueError(
                f'sender_type {sender_type} not one of customer, agent, action')
        parts.append(start)
        parts.append(utterance)
        parts.append(end)

    return separator.join(parts)

def convo_to_context_response_pairs_abcd(
    utterance_rows: List[List[str]],
):
    context_response_pairs: List[Dict] = []
    for idx, row in enumerate(utterance_rows):
        sender_type, utterance = row
        if (sender_type == 'agent'):
            context = convo_to_text_abcd(utterance_rows[:idx])
            context_response_pairs.append({'context': context, 'response': utterance})
    
    df_context_response_pairs = pd.DataFrame(context_response_pairs)
    
    return df_context_response_pairs

def convo_to_text_multi_woz(
    example: Dict,
    include_start: bool = True,
    separator: str = "",
    convo_start: str = CONVO_START,
    convo_end: str = CONVO_END,
    cus_start: str = CUS_START,
    cus_end: str = CUS_END,
    rep_start: str = REP_START,
    rep_end: str = REP_END,
):
    parts: List[str] = []
    if include_start:
        parts.append(convo_start)

    for utterance, sender_idx in zip(example['turns']['utterance'], example['turns']['speaker']):
        if sender_idx == 0:
            start, end = cus_start, cus_end
        elif sender_idx == 1:
            start, end = rep_start, rep_end
        else:
            raise ValueError(
                f'{sender_idx=} not one of 0, 1')
        parts.append(start)
        parts.append(utterance)
        parts.append(end)

    return separator.join(parts)


def convo_to_context_response_pairs_multi_woz(
    example: Dict,
):
    context_response_pairs: List[Dict] = []
    context = CONVO_START
    for utterance, sender_idx in zip(example['turns']['utterance'], example['turns']['speaker']):
        if sender_idx == 0: # customer
            context += CUS_START + utterance + CUS_END
        elif sender_idx == 1: # agent
            context_response_pairs.append({'context': context, 'response': utterance})
            context += REP_START + utterance + REP_END
        else:
            raise ValueError(
                f'{sender_idx=} not one of 0, 1')
    
    df_context_response_pairs = pd.DataFrame(context_response_pairs)
    
    return df_context_response_pairs


def convo_to_text_taskmaster3(
    example: Dict,
    include_start: bool = True,
    separator: str = "",
    convo_start: str = CONVO_START,
    convo_end: str = CONVO_END,
    cus_start: str = CUS_START,
    cus_end: str = CUS_END,
    rep_start: str = REP_START,
    rep_end: str = REP_END,
):
    parts: List[str] = []
    if include_start:
        parts.append(convo_start)

    for utt_dict in example['utterances']:
        if utt_dict['speaker'] == 'user':
            start, end = cus_start, cus_end
        elif utt_dict['speaker'] == 'assistant':
            start, end = rep_start, rep_end
        else:
            raise ValueError(
                f"{utt_dict['speaker']=} not one of 0, 1")
        parts.append(start)
        parts.append(utt_dict['text'])
        parts.append(end)

    return separator.join(parts)


def convo_to_context_response_pairs_taskmaster3(
    example: Dict,
):
    context_response_pairs: List[Dict] = []
    context = CONVO_START
    for utt_dict in example['utterances']:
        utterance = utt_dict['text']
        if utt_dict['speaker'] == 'user': # customer
            context += CUS_START + utterance + CUS_END
        elif utt_dict['speaker'] == 'assistant': # agent
            context_response_pairs.append({'context': context, 'response': utterance})
            context += REP_START + utterance + REP_END
        else:
            raise ValueError(
                f"{utt_dict['speaker']=} not one of 0, 1")
    
    df_context_response_pairs = pd.DataFrame(context_response_pairs)
    
    return df_context_response_pairs