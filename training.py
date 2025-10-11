import torch
import random

def create_training_instance(sent_A, sent_B, is_next_label, tokenizer, max_len):
    """
    Creates a single training instance for BERT pre-training.
    
    Args:
        sent_A (str): The first sentence.
        sent_B (str): The second sentence.
        is_next_label (int): 0 if B is the next sentence, 1 if it's a random sentence.
        tokenizer: A tokenizer object with vocab, encode(), and special tokens.
        max_len (int): The maximum sequence length.
    """
    
    # Special tokens from the tokenizer
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    mask_token = tokenizer.mask_token
    pad_token = tokenizer.pad_token

    # 1. Tokenize and format
    tokens_a = tokenizer.tokenize(sent_A)
    tokens_b = tokenizer.tokenize(sent_B)
    
    # Truncate to make sure the combined sequence fits within max_len
    # -3 for [CLS], [SEP], [SEP]
    _truncate_seq_pair(tokens_a, tokens_b, max_len - 3)
    
    # Combine tokens
    tokens = [cls_token] + tokens_a + [sep_token] + tokens_b + [sep_token]
    
    # Generate segment IDs
    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    
    # 2. Perform masking for MLM
    mlm_labels = [-100] * len(tokens) # -100 is ignored by PyTorch CrossEntropyLoss
    candidate_indices = []
    
    # Find all indices that can be masked (not special tokens)
    for i, token in enumerate(tokens):
        if token not in [cls_token, sep_token]:
            candidate_indices.append(i)

    # Number of tokens to mask (15%)
    num_to_mask = max(1, int(round(len(candidate_indices) * 0.15)))
    random.shuffle(candidate_indices)
    masked_indices = sorted(candidate_indices[:num_to_mask])

    for i in masked_indices:
        # Store the original token ID as the label
        mlm_labels[i] = tokenizer.convert_tokens_to_ids([tokens[i]])[0]
        
        # Apply 80/10/10 masking strategy
        rand = random.random()
        if rand < 0.8:
            tokens[i] = mask_token
        elif rand < 0.9:
            # Replace with a random token
            random_token_id = random.randint(0, tokenizer.vocab_size - 1)
            tokens[i] = tokenizer.convert_ids_to_tokens([random_token_id])[0]
        # else (10%): keep the original token
            
    # 3. Generate final tensors
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Padding
    padding_len = max_len - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_len
    segment_ids += [0] * padding_len
    mlm_labels += [-100] * padding_len
    
    # Attention mask (1 for real tokens, 0 for padding)
    attention_mask = [1] * (len(tokens)) + [0] * padding_len
    
    # Convert to PyTorch tensors
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "token_type_ids": torch.tensor(segment_ids, dtype=torch.long), # Segment IDs
        "nsp_label": torch.tensor(is_next_label, dtype=torch.long),
        "mlm_labels": torch.tensor(mlm_labels, dtype=torch.long)
    }

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

# --- Example Usage ---
# You would need a real tokenizer for this to run
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# max_sequence_length = 128
#
# # Case 1: IsNext
# sent_A1 = "The cat sat on the mat."
# sent_B1 = "It was very comfortable."
# instance1 = create_training_instance(sent_A1, sent_B1, 0, tokenizer, max_sequence_length)
#
# # Case 2: NotNext
# sent_A2 = "The sun is shining brightly."
# sent_B2 = "My favorite programming language is Python." # Random sentence
# instance2 = create_training_instance(sent_A2, sent_B2, 1, tokenizer, max_sequence_length)
#
# print("--- Instance 1 (IsNext) ---")
# for key, value in instance1.items():
#     print(f"{key}: {value.shape}")
#
# print("\n--- Instance 2 (NotNext) ---")
# for key, value in instance2.items():
#     print(f"{key}: {value.shape}")