import torch
torch.manual_seed(0)
from torch.utils.data import Dataset
# from torchvision.transforms import ToTensor, Lambda
import numpy as np
from torch.utils.data import DataLoader, random_split
from utils import calculate_class_weights
import pandas as pd
import random
import yaml
from utils import *
from Bio.Seq import Seq


class LocalizationDataset(Dataset):
    def __init__(self, samples, configs,mode="train"):
        # self.label_to_index = {"Other": 0, "SP": 1, "MT": 2, "CH": 3, "TH": 4}
        # self.index_to_label = {0: "Other", 1: "SP", 2: "MT", 3: "CH", 4: "TH"}
        # self.transform = transform
        # self.target_transform = target_transform
        # self.cs_transform = cs_transform
        self.original_samples = samples
        self.n = configs.encoder.num_classes
        self.class_weights = calculate_class_weights(self.count_samples_by_class(self.n, self.original_samples))
        print(self.class_weights)
        if mode == "train" and configs.train_settings.data_aug.enable:
           self.data_aug = True
           samples = self.data_aug_train(samples,configs,self.class_weights)

        self.samples = samples
        #print(samples[0:2]) #same as original
        self.apply_supcon = configs.supcon.apply
        if self.apply_supcon:
           self.n_pos = configs.supcon.n_pos
           self.n_neg = configs.supcon.n_neg
           self.hard_neg = configs.supcon.hard_neg

        self.mode = mode

    #"""
    def random_mutation(self, sequence, target, pos_mutation_rate, neg_mutation_rate):
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # List of standard amino acids
        seq = Seq(sequence)
        seq_list = list(seq)
        # Get the mutable positions
        # print(target)
        pos_mutable_positions = [i for i, label in enumerate(target) if label == 1]
        if len(pos_mutable_positions) == 1:
            num_pos_mutations = 1 if random.random() < pos_mutation_rate else 0
        else:
            num_pos_mutations = int(pos_mutation_rate * len(pos_mutable_positions))

        neg_mutable_positions = [i for i, label in enumerate(target) if label == 0]
        num_neg_mutations = int(neg_mutation_rate * len(neg_mutable_positions))
        if num_pos_mutations > 0 or num_neg_mutations > 0:
            if num_pos_mutations > 0:
                num_pos_mutations = min(num_pos_mutations, len(pos_mutable_positions))
                mutation_positions = random.sample(pos_mutable_positions, num_pos_mutations)
                for pos in mutation_positions:
                    # Ensure the mutated amino acid is different from the original
                    new_aa = random.choice([aa for aa in amino_acids if aa != seq_list[pos]])
                    seq_list[pos] = new_aa

            if num_neg_mutations > 0:
                num_neg_mutations = min(num_neg_mutations, len(neg_mutable_positions))
                mutation_positions = random.sample(neg_mutable_positions, num_neg_mutations)
                for pos in mutation_positions:
                    # Ensure the mutated amino acid is different from the original
                    new_aa = random.choice([aa for aa in amino_acids if aa != seq_list[pos]])
                    seq_list[pos] = new_aa

            # Join the mutated amino acids back into a sequence
            mutated_sequence = ''.join(seq_list)
            # print(sequence)
            # print(mutated_sequence)
            return mutated_sequence
        else:
            return sequence
    #"""

    def data_aug_train(self, samples, configs, class_weights):
        print("data aug on len of " + str(len(samples)))
        aug_samples = []
        # aug_samples_tag = []
        pos_mutation_rate, neg_mutation_rate = configs.train_settings.data_aug.pos_mutation_rate, configs.train_settings.data_aug.neg_mutation_rate

        # if 1 < 0:
        #     for id, id_frag_list, seq_frag_list, target_frag_list, type_protein in samples:
        #         aug_samples.append((id, id_frag_list, seq_frag_list, target_frag_list, type_protein))  # add original

        for id, id_frag_list, seq_frag_list, target_frag_list, type_protein in samples:
            if configs.train_settings.data_aug.add_original:
                if configs.train_settings.data_aug.add_original_but_not_export:
                    if type_protein[4] != 1:  # 0619 yichuan if not export
                        # print(type_protein[4])
                        aug_samples.append((id, id_frag_list, seq_frag_list, target_frag_list, type_protein, 'ORI'))
                        # aug_samples_tag.append(1)
                else:
                    aug_samples.append((id, id_frag_list, seq_frag_list, target_frag_list, type_protein, 'ORI'))  # add original
                    # aug_samples_tag.append(1)

            class_positions = np.where(type_protein == 1)[0]

            # 这里我改了, 为了测试
            per_times = np.max([1, int(np.ceil(
                configs.train_settings.data_aug.per_times * np.max([class_weights[x] for x in class_positions])))])
            # per_times = 1


            temp_target_frag_list = target_frag_list.copy()
            for aug_i in range(per_times):
                aug_id = id + "_" + str(aug_i)
                aug_id_frag_list = [aug_id + "@" + id_frag.split("@")[1] for id_frag in id_frag_list]

                if aug_i == 0:
                    flattened_aug_target_frag_list = np.hstack(temp_target_frag_list)

                    if 1 in flattened_aug_target_frag_list[0] or 1 in flattened_aug_target_frag_list[4]:
                        pass

                    if 1 in flattened_aug_target_frag_list[1]:
                        # pass
                        stop_left = flattened_aug_target_frag_list[1].tolist().index(1)
                        flattened_aug_target_frag_list[1][stop_left + 1:] = [1] * \
                                                                      (len(flattened_aug_target_frag_list[1]) - stop_left - 1)

                    if 1 in flattened_aug_target_frag_list[2]:
                        # pass
                        stop_left = flattened_aug_target_frag_list[2].tolist().index(1)
                        stop_right = len(flattened_aug_target_frag_list[2]) - 1 - \
                                     flattened_aug_target_frag_list[2].tolist()[::-1].index(1)
                        stop = (stop_left+stop_right)/2
                        if stop < len(flattened_aug_target_frag_list[2]) / 2:
                            flattened_aug_target_frag_list[2][:stop_right] = [1] * stop_right
                        else:
                            flattened_aug_target_frag_list[2][stop_left + 1:] = [1] * (
                                        len(flattened_aug_target_frag_list[2]) - stop_left - 1)

                    N_side = [3, 5, 6, 7]
                    for idx in N_side:
                        # pass
                        if 1 in flattened_aug_target_frag_list[idx]:
                            stop_right = len(flattened_aug_target_frag_list[idx]) - 1 - \
                                         flattened_aug_target_frag_list[idx].tolist()[::-1].index(1)
                            flattened_aug_target_frag_list[idx][:stop_right] = [1] * stop_right

                    shapes = [arr.shape for arr in temp_target_frag_list]

                    split_indices = np.cumsum([shape[1] for shape in shapes])[:-1]

                    temp_target_frag_list = np.split(flattened_aug_target_frag_list, split_indices, axis=1)

                # if len(seq_frag_list) == 2:
                #     for sequence, target in zip(seq_frag_list, aug_target_frag_list):
                #         print(sequence)
                #         print(target)
                #         print([int(max(set(column))) for column in zip(*target)][:len(sequence)])
                #         print("!!!")
                #     exit(0)

                aug_seq_frag_list = [
                    self.random_mutation(sequence, [int(max(set(column))) for column in zip(*target)][:len(sequence)],
                                         pos_mutation_rate, neg_mutation_rate) for sequence, target in
                    zip(seq_frag_list, temp_target_frag_list)]

                # aug_seq_frag_list = [
                #     self.random_mutation(sequence, [int(max(set(column))) for column in zip(*target)][:len(sequence)],
                #                          configs.train_settings.data_aug.mutation_rate) for sequence, target in
                #     zip(seq_frag_list, target_frag_list)]

                aug_target_frag_list = target_frag_list
                aug_type_protein = type_protein
                aug_samples.append(
                    (aug_id, aug_id_frag_list, aug_seq_frag_list, aug_target_frag_list, aug_type_protein, 'AUG'))
                # aug_samples_tag.append(0)

            # print(aug_type_protein)
            # print(aug_seq_frag_list)
            # print(target_frag_list)
            # print(aug_target_frag_list)
            # print()
            # print()
        # print(len(aug_samples_tag))
        print("data length after augmentation is ", str(len(aug_samples)))
        # exit(0)
        return aug_samples

    @staticmethod
    def count_samples_by_class(n, samples):
        """Count the number of samples for each class."""
        # class_counts = {}
        class_counts = np.zeros(n)  # one extra is for samples without motif
        # Iterate over the samples
        for id, id_frag_list, seq_frag_list, target_frag_list, type_protein in samples:
            class_counts += type_protein
        return class_counts

    def __len__(self):
           return len(self.samples)

    def __getitem__(self, idx):
        #print(idx)
        if self.mode == "train":
            id, id_frag_list, seq_frag_list, target_frag_list, type_protein, ORI_AUG = self.samples[idx]
        else:
            id, id_frag_list, seq_frag_list, target_frag_list, type_protein = self.samples[idx]
            ORI_AUG = None
        labels = np.where(type_protein == 1)[0]
        weights = []
        for label in labels:
            weights.append(self.class_weights[label])
        sample_weight = max(weights)
        # print(self.class_weights)
        # exit(0) 0613 check
        # labels=np.where(np.max(target_frags, axis=1)==1)[0]
        # weights=[]
        # for label in labels:
        #     weights.append(self.class_weights[label])
        # if np.max(target_frags)==0:
        #     weights.append(self.class_weights[self.n])

        # sample_weight = max(weights)
        # target_frags = torch.from_numpy(np.stack(target_frags, axis=0))
        type_protein = torch.from_numpy(type_protein)
        pos_neg = None
        if self.apply_supcon:
            # Even when not in warm starting, the following code is still executed, although its results are not used
            pos_samples = self.get_pos_samples(idx)
            neg_samples = self.get_neg_samples(idx)
            pos_neg = [pos_samples, neg_samples]
        return id, id_frag_list, seq_frag_list, target_frag_list, type_protein, sample_weight, pos_neg, ORI_AUG
        # return id, type_protein

    def get_pos_samples(self, anchor_idx):
        filtered_samples = [sample for idx, sample in enumerate(self.samples) if idx != anchor_idx] #all candidate exlude itself.
        anchor_type_protein = self.samples[anchor_idx][4] #class 0000 0001
        pos_samples = [sample for sample in filtered_samples if
                       np.any(np.logical_and(anchor_type_protein == 1, sample[4] == 1))]
        if len(pos_samples) < self.n_pos:
            # raise ValueError(f"Not enough positive samples for {anchor_type_protein} found: {len(pos_samples)}. Required: {self.n_pos}.")
            samples_to_add = self.n_pos - len(pos_samples)
            for _ in range(samples_to_add):
                pos_samples.append(random.choice(pos_samples))
        if len(pos_samples) > self.n_pos:
            pos_samples = random.sample(pos_samples, self.n_pos)

        pos_samples_with_weight = []
        for sample in pos_samples:
            labels = np.where(sample[4] == 1)[0]
            weights = [self.class_weights[label] for label in labels]
            sample_weight = np.max(weights)
            sample_with_weight = list(sample) + [sample_weight]
            pos_samples_with_weight.append(sample_with_weight)

        return pos_samples_with_weight  # pos_samples

    def get_neg_samples(self, anchor_idx):
        filtered_samples = [sample for idx, sample in enumerate(self.samples) if idx != anchor_idx]
        anchor_type_protein = self.samples[anchor_idx][4] #class
        if self.hard_neg:
            hneg = self.hard_mining(anchor_type_protein) #similiar
            neg_samples = [sample for sample in filtered_samples if
                           np.any(np.logical_and(hneg == 1, sample[4] == 1))]
        else:
            neg_samples = [sample for sample in filtered_samples if
                           not np.any(np.logical_and(anchor_type_protein == 1, sample[4] == 1))]
        if len(neg_samples) < self.n_neg:
            # raise ValueError(f"Not enough negative samples ({hneg}) for {anchor_type_protein} found: {len(neg_samples)}. Required: {self.n_neg}.")
            # print(f"Not enough negative samples ({hneg}) for {anchor_type_protein} found: {len(neg_samples)}. Required: {self.n_neg}.")
            # print('The rest of negative samples are randomly selected.')
            neg_samples_2 = [sample for sample in filtered_samples if
                           not np.any(np.logical_and(anchor_type_protein == 1, sample[4] == 1))]
            neg_samples_2 = random.sample(neg_samples_2, self.n_neg-len(neg_samples))
            neg_samples.extend(neg_samples_2)
        if len(neg_samples) > self.n_neg:
            neg_samples = random.sample(neg_samples, self.n_neg)

        neg_samples_with_weight = []
        for sample in neg_samples:
            labels = np.where(sample[4] == 1)[0]
            weights = [self.class_weights[label] for label in labels]
            sample_weight = np.max(weights)
            sample_with_weight = list(sample) + [sample_weight]
            neg_samples_with_weight.append(sample_with_weight)

        return neg_samples_with_weight  # neg_samples

    @staticmethod
    def hard_mining(anchor_type_protein, file_path='distance_map.txt'):
        with open(file_path, 'r') as source_file:
            content = eval(source_file.read())

        # print(content)
        min_non_zero_keys = {}

        for main_key, sub_dict in content.items():
            # Filter out zero values and find the minimum
            non_zero_values = {k: v for k, v in sub_dict.items() if v > 0}
            min_key = min(non_zero_values, key=non_zero_values.get)
            min_non_zero_keys[main_key] = min_key

        label2idx = {"Nucleus": 0, "ER": 1, "Peroxisome": 2, "Mitochondrion": 3, "Nucleus_export": 4,
                     "SIGNAL": 5, "chloroplast": 6, "Thylakoid": 7}
        mapped_dict = {label2idx[key]: label2idx[value] for key, value in min_non_zero_keys.items()}

        neg = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(anchor_type_protein)):
            if anchor_type_protein[i] == 1:
                neg[mapped_dict[i]] = 1
        for i in range(len(anchor_type_protein)):
            if anchor_type_protein[i] == 1:
                neg[i] = 0
        if all(v == 0 for v in neg):
            neg = [1 if x == 0 else 0 for x in anchor_type_protein]
        return np.array(neg)

def custom_collate(batch):
    id, id_frags, fragments, target_frags, type_protein, sample_weight, pos_neg, ORI_AUG = zip(*batch)
    return id, id_frags, fragments, target_frags, type_protein, sample_weight, pos_neg, ORI_AUG


def prot_id_to_seq(seq_file):
    id2seq = {}
    with open(seq_file) as file:
        for line in file:
            id = line.strip().split("\t")[0]
            seq = line.strip().split("\t")[2]
            id2seq[id] = seq
    return id2seq

# def prepare_samples(exclude, fold_num_list, id2seq_dic, npz_file):
#     samples = []
#     data = np.load(npz_file)
#     fold = data['fold']
#     if exclude:
#         index = [all(num != target for target in fold_num_list) for num in fold]
#         # index = fold != fold_num
#     else:
#         index = [any(num == target for target in fold_num_list) for num in fold]
#         # index = fold == fold_num
#     prot_ids = data['ids'][index]
#     y_type = data['y_type'][index]
#     y_cs = data['y_cs'][index]
#     for idx, prot_id in enumerate(prot_ids):
#         seq = id2seq_dic[prot_id]
#         # if len(seq)>200:
#         #   seq=seq[:200]
#         label = y_type[idx]
#         position = np.argmax(y_cs[idx])
#         samples.append((seq, int(label), position))
#     return samples

def split_protein_sequence(prot_id, sequence, targets, configs):
    fragment_length = configs.encoder.max_len - 2
    overlap = configs.encoder.frag_overlap
    fragments = []
    target_frags = []
    id_frags = []
    sequence_length = len(sequence)
    start = 0
    ind = 0

    while start < sequence_length:
        end = start + fragment_length
        if end > sequence_length:
            end = sequence_length
        fragment = sequence[start:end]
        target_frag = targets[:, start:end]
        if target_frag.shape[1] < fragment_length:
            pad = np.zeros([targets.shape[0], fragment_length-target_frag.shape[1]])
            target_frag = np.concatenate((target_frag, pad), axis=1)
        target_frags.append(target_frag)
        fragments.append(fragment)
        id_frags.append(prot_id+"@"+str(ind))
        ind += 1
        if start + fragment_length > sequence_length:
            break
        start += fragment_length - overlap

    return id_frags, fragments, target_frags


#["Nucleus","ER","Peroxisome","Mitochondrion","Nucleus_export","dual","SIGNAL","chloroplast","Thylakoid"]
def fix_sample(motif_left, motif_right, label, label2idx, type_protein, targets):
    if motif_left == "None":
        motif_left = 0
    else:
        motif_left = int(motif_left)-1
    motif_right = int(motif_right)
    if label == "Thylakoid" and motif_left != 0:
        index_row = label2idx["chloroplast"]
        type_protein[index_row] = 1
        targets[index_row, motif_left-1] = 1
    return motif_left, motif_right, type_protein, targets


def prepare_samples(csv_file, configs):
    # label2idx = {"Nucleus":0, "ER":1, "Peroxisome":2, "Mitochondrion":3, "Nucleus_export":4,
    #              "dual":5, "SIGNAL":6, "chloroplast":7, "Thylakoid":8}
    label2idx = {"Nucleus": 0, "ER": 1, "Peroxisome": 2, "Mitochondrion": 3, "Nucleus_export": 4,
                 "SIGNAL": 5, "chloroplast": 6, "Thylakoid": 7}
    samples = []
    n = configs.encoder.num_classes
    df = pd.read_csv(csv_file)
    row, col = df.shape
    for i in range(row):
        prot_id = df.loc[i, "Entry"]
        seq = df.loc[i, "Sequence"]
        targets = np.zeros([n, len(seq)])
        type_protein = np.zeros(n)
        # motifs = df.iloc[i,1:-2]
        motifs = df.loc[i, "MOTIF"].split("|")
        for motif in motifs:
            if not pd.isnull(motif):
                # label = motif.split("|")[0].split(":")[1]
                label = motif.split(":")[1]
                # motif_left = motif.split("|")[0].split(":")[0].split("-")[0]
                motif_left = motif.split(":")[0].split("-")[0]
                motif_right = motif.split(":")[0].split("-")[1]

                motif_left, motif_right, type_protein, targets = fix_sample(motif_left, motif_right, label, label2idx, type_protein, targets)
                if label in label2idx:
                    index_row = label2idx[label]
                    type_protein[index_row] = 1
                    if label in ["SIGNAL", "chloroplast", "Thylakoid", "Mitochondrion"]:
                        targets[index_row, motif_right-1] = 1
                    elif label == "Peroxisome" and motif_left == 0:
                        targets[index_row, motif_right-1] = 1
                    elif label == "Peroxisome" and motif_left != 0:
                        targets[index_row, motif_left] = 1
                    elif label == "ER":
                        targets[index_row, motif_left] = 1
                    elif label == "Nucleus" or label == "Nucleus_export":
                        targets[index_row, motif_left:motif_right] = 1

        id_frag_list, seq_frag_list, target_frag_list = split_protein_sequence(prot_id, seq, targets, configs)
        samples.append((prot_id, id_frag_list, seq_frag_list, target_frag_list, type_protein))
        # for j in range(len(fragments)):
        #     id=prot_id+"@"+str(j)
        #     samples.append((id, fragments[j], target_frags[j], type_protein))
    return samples


def prepare_dataloaders(configs, valid_batch_number, test_batch_number):
    # id_to_seq = prot_id_to_seq(seq_file)
    if configs.train_settings.dataset == 'v2':
        samples = prepare_samples("./parsed_EC7_v2/PLANTS_uniprot.csv", configs)
        samples.extend(prepare_samples("./parsed_EC7_v2/ANIMALS_uniprot.csv", configs))
        samples.extend(prepare_samples("./parsed_EC7_v2/FUNGI_uniprot.csv", configs))
        cv = pd.read_csv("./parsed_EC7_v2/split/type/partition.csv")
    elif configs.train_settings.dataset == 'v2_cdhit0.9':
        print("v2_cdhit0.9")
        samples = prepare_samples("./parsed_EC7_v2_cdhit0.9/PLANTS_uniprot_cdhit.csv", configs)
        samples.extend(prepare_samples("./parsed_EC7_v2_cdhit0.9/ANIMALS_uniprot_cdhit.csv", configs))
        samples.extend(prepare_samples("./parsed_EC7_v2_cdhit0.9/FUNGI_uniprot_cdhit.csv", configs))
        cv = pd.read_csv("./parsed_EC7_v2/split/type/partition.csv")
    elif configs.train_settings.dataset == 'v2_cdhit0.9_signal':
        print("v2_cdhit0.9_signal")
        samples = prepare_samples("./parsed_EC7_v2_cdhit0.9/PLANTS_uniprot_cdhit_signal.csv", configs)
        samples.extend(prepare_samples("./parsed_EC7_v2_cdhit0.9/ANIMALS_uniprot_cdhit_signal.csv", configs))
        samples.extend(prepare_samples("./parsed_EC7_v2_cdhit0.9/FUNGI_uniprot_cdhit_signal.csv", configs))
        cv = pd.read_csv("./parsed_EC7_v2/split/type/partition.csv")
    elif configs.train_settings.dataset == 'v3':
        samples = prepare_samples("./parsed_EC7_v3/PLANTS_uniprot.csv", configs)
        samples.extend(prepare_samples("./parsed_EC7_v3/ANIMALS_uniprot.csv", configs))
        samples.extend(prepare_samples("./parsed_EC7_v3/FUNGI_uniprot.csv", configs))
        cv = pd.read_csv("./parsed_EC7_v3/split/type/partition.csv")
    elif configs.train_settings.dataset == 'v4':
        samples = prepare_samples("./parsed_v4/PLANTS_uniprot.csv", configs)
        samples.extend(prepare_samples("./parsed_v4/ANIMALS_uniprot.csv", configs))
        samples.extend(prepare_samples("./parsed_v4/FUNGI_uniprot.csv", configs))
        cv = pd.read_csv("./parsed_v4/partition.csv")

    train_id = []
    val_id = []
    test_id = []
    id = cv.loc[:, 'entry']
    # split=cv.loc[:,'split']
    # fold=cv.loc[:,'fold']
    partition = cv.loc[:, 'partition']
    for i in range(len(id)):
        # f=fold[i]
        # s=split[i]
        p = partition[i]
        d = id[i]
        if p == valid_batch_number:
            val_id.append(d)
        elif p == test_batch_number:
            test_id.append(d)
        else:
            train_id.append(d)


    # print(train_id)


    train_sample = []
    valid_sample = []
    test_sample = []
    for i in samples:
        # id=i[0].split("@")[0]
        id = i[0]
        # print(id)
        if id in train_id:
            train_sample.append(i)
        elif id in val_id:
            valid_sample.append(i)
        elif id in test_id:
            test_sample.append(i)

    # train_samples = prepare_samples(exclude=True, fold_num_list=[valid_batch_number, test_batch_number], id2seq_dic=id_to_seq, npz_file=npz_file)
    # valid_samples = prepare_samples(exclude=False, fold_num_list=[valid_batch_number], id2seq_dic=id_to_seq, npz_file=npz_file)
    # test_samples = prepare_samples(exclude=False, fold_num_list=[test_batch_number], id2seq_dic=id_to_seq, npz_file=npz_file)
    #print("first train[0]") #same!
    #print(train_sample[0])
    random.seed(configs.fix_seed)
    # Shuffle the list
    random.shuffle(samples)
    # train_dataset = LocalizationDataset(samples, configs=configs)
    # dataset_size = len(train_dataset)
    # train_size = int(0.8 * dataset_size)  # 80% for training, adjust as needed
    # test_size = dataset_size - train_size
    # train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])
    # val_size = train_size - int(0.8 * train_size)
    # train_dataset, valid_dataset = random_split(train_dataset, [int(0.8 * train_size), val_size])



    # print(train_dataset)
    #print(train_sample[0])
    train_dataset = LocalizationDataset(train_sample, configs=configs,mode = "train")
    valid_dataset = LocalizationDataset(valid_sample, configs=configs,mode = "valid")
    test_dataset = LocalizationDataset(test_sample, configs=configs,mode = "test")
    train_dataloader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size, shuffle=True, collate_fn=custom_collate,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=configs.valid_settings.batch_size, shuffle=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.valid_settings.batch_size, shuffle=False, collate_fn=custom_collate)
    return {'train': train_dataloader, 'test': test_dataloader, 'valid': valid_dataloader}


if __name__ == '__main__':
    config_path = '0530config/config_nosupcon.yaml'
    with open(config_path) as file:
        configs_dict = yaml.full_load(file)

    configs_file = load_configs(configs_dict)

    dataloaders_dict = prepare_dataloaders(configs_file, 0, 1)
    train_loader=dataloaders_dict["train"]
    for batch in train_loader:
        # id_batch, fragments_batch, target_frags_batch, weights_batch = batch
        (prot_id, id_frag_list, seq_frag_list, target_frag_nplist, type_protein_pt, sample_weight,pos_neg) = batch
        # id, type_protein = batch
        # print(len(id_batch))
        # print(len(fragments_batch))
        # print(np.array(target_frags_batch).shape)
        # print(len(weights_batch))
        print("==========================")
        print("==========================")
        #print(type(prot_id))
        print("prot_id="+str(prot_id))
        #print(pos_neg)
        print("id_frag_list="+str(id_frag_list))
        print("seq_frag_list="+str(seq_frag_list))
        #print("target_frag_nplist="+str(target_frag_nplist))
        print(target_frag_nplist[0])
        print([len(frag) for frag in id_frag_list])
        #print(type(id_frag_list))
        #print(id_frag_list)
        #print(type(seq_frag_list))
        #print(seq_frag_list)
        #print(type(target_frag_nplist))
        #print(target_frag_nplist)
        #print(type(type_protein_pt))
        #print(type_protein_pt)
        #print(type(sample_weight))
        #print(sample_weight)
        ## print(next(iter(dataloaders_dict['test'])))
        ## a=np.array(target_frags_batch)
        ## print(np.max(a, axis=2))
        ## print(target_frags_batch.size())
        ## print(target_frags_batch[1])
        ## print(weights_batch)
        break

    print('done')

