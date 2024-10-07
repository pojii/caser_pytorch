import numpy as np
import scipy.sparse as sp

class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions. This is designed only for implicit feedback scenarios.

    Parameters
    ----------

    file_path: file contains (user,item,rating) triplets
    user_map: dict of user mapping
    item_map: dict of item mapping
    """

    def __init__(self, file_path,
                 user_map=None,
                 item_map=None):

        if not user_map or not item_map:
            user_map = dict()
            item_map = dict()

            num_user = 0
            num_item = 0
        else:
            num_user = len(user_map)
            num_item = len(item_map)

        user_ids = list()
        item_ids = list()
        
        with open(file_path, 'r') as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) == 2:
                    u, i = parts
                    user_ids.append(u)
                    item_ids.append(i)
                elif len(parts) == 3:
                    u, i, r = parts
                    if r == '1':  # เก็บเฉพาะกรณีที่มีการดู
                        user_ids.append(u)
                        item_ids.append(i)
                else:
                    print(f"Skipping line with unexpected format: {line.strip()}")

        # update user and item mapping
        for u in user_ids:
            if u not in user_map:
                user_map[u] = num_user
                num_user += 1
        for i in item_ids:
            if i not in item_map:
                item_map[i] = num_item
                num_item += 1

        user_ids = np.array([user_map[u] for u in user_ids])
        item_ids = np.array([item_map[i] for i in item_ids])

        self.num_users = num_user
        self.num_items = num_item

        self.user_ids = user_ids
        self.item_ids = item_ids

        self.user_map = user_map
        self.item_map = item_map

        self.sequences = None
        self.test_sequences = None

    def __len__(self):
        return len(self.user_ids)

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """
        row = self.user_ids
        col = self.item_ids
        data = np.ones(len(self))
        return sp.csr_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def to_sequence(self, sequence_length=5, target_length=10):
        max_sequence_length = sequence_length + target_length

        # Sort first by user id
        sort_indices = np.lexsort((self.user_ids,))
        # print('self.user_ids', self.user_ids)
        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]
        # print('item_ids', item_ids)
        user_ids, indices, counts = np.unique(user_ids,
                                            return_index=True,
                                            return_counts=True)
        # print('user_ids, indices, counts', user_ids, indices, counts)
        # print('counts',counts)
        temp_sum = [max(1, c - sequence_length) for c in counts]
        # print('temp_sum', temp_sum)
        num_subsequences = sum(temp_sum)
        # for i in range(len(temp_sum)):
        #     print(f"User {i} has {temp_sum[i]} subsequences")
        #     print(f"User {i} has {counts[i]} items")
        #     print(f"User {i} has {counts[i] - sequence_length} subsequences")
        # print('num_subsequences', num_subsequences)
        sequences = np.zeros((num_subsequences, sequence_length), dtype=np.int64)
        # for i in range(len(temp_sum)):
        #     print(f"User {i} has {sequences[i]} subsequences")
        sequences_targets = np.zeros((num_subsequences, target_length), dtype=np.int64)
        sequence_users = np.empty(num_subsequences, dtype=np.int64)

        test_sequences = np.zeros((self.num_users, sequence_length), dtype=np.int64)
        test_users = np.empty(self.num_users, dtype=np.int64)

        idx = 0
        for user, user_item_ids in zip(user_ids, np.split(item_ids, indices[1:])):
            if len(user_item_ids) >= 2:
                if len(user_item_ids) > sequence_length:
                    for i in range(len(user_item_ids) - sequence_length):
                        seq = user_item_ids[i:i + sequence_length]
                        tgt = user_item_ids[i + sequence_length:i + 2 * sequence_length]
                else:
                    half_length = len(user_item_ids) // 2
                    seq = user_item_ids[:half_length]
                    tgt = user_item_ids[half_length:]
                    
                # Left-pad sequences with 0
                sequences[idx, -len(seq):] = seq
                # Right-pad targets with 0
                sequences_targets[idx, :len(tgt)] = tgt
                sequence_users[idx] = user
                idx += 1
            
            # Test sequences (last sequence for the user)
            test_sequences[user, -len(user_item_ids[-sequence_length:]):] = user_item_ids[-sequence_length:]
            test_users[user] = user

        # Remove sequences that are all zeros
        non_zero_indices = [i for i in range(len(sequences)) if not np.all(sequences[i] == 0)]
        sequences = sequences[non_zero_indices]
        sequences_targets = sequences_targets[non_zero_indices]
        sequence_users = sequence_users[non_zero_indices]

        # Sample sequences if they exceed max_samples
        if len(sequences) > 100000:
            indices = np.random.choice(len(sequences), 100000, replace=False)
            sequences = np.array([sequences[i] for i in indices])
            sequences_targets = np.array([sequences_targets[i] for i in indices])
            sequence_users = np.array([sequence_users[i] for i in indices])

        print(f"Total sequences: {len(sequences)}")

        print(f"Total sequences: {len(sequences)}")

        # Save to files
        np.savetxt('sequences.txt', sequences, fmt='%d', delimiter=',')
        np.savetxt('sequences_targets.txt', sequences_targets, fmt='%d', delimiter=',')
        np.savetxt('sequence_users.txt', sequence_users, fmt='%d')
        np.savetxt('test_sequences.txt', test_sequences, fmt='%d', delimiter=',')

        self.sequences = SequenceInteractions(sequence_users, sequences, sequences_targets)
        self.test_sequences = SequenceInteractions(test_users, test_sequences)


class SequenceInteractions(object):
    """
    Interactions encoded as a sequence matrix.

    Parameters
    ----------
    user_ids: np.array
        sequence users
    sequences: np.array
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    targets: np.array
        sequence targets
    """

    def __init__(self,
                 user_ids,
                 sequences,
                 targets=None):
        self.user_ids = user_ids
        self.sequences = sequences
        self.targets = targets

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]

def _generate_sequences(user_ids, item_ids,
                        indices,
                        max_sequence_length):
    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq in _sliding_window(item_ids[start_idx:stop_idx],
                                   max_sequence_length):
            yield (user_ids[i], seq)

def _sliding_window(tensor, window_size, step_size=1):
    if len(tensor) - window_size >= 0:
        for i in range(len(tensor), 0, -step_size):
            if i - window_size >= 0:
                yield tensor[i - window_size:i]
            else:
                break
    else:
        num_paddings = window_size - len(tensor)
        # Pad sequence with 0s if it is shorter than windows size.
        yield np.pad(tensor, (num_paddings, 0), 'constant')