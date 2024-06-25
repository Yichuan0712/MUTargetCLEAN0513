import torch
torch.manual_seed(0)
from torch.cuda.amp import GradScaler, autocast
import argparse
import os
import yaml
import numpy as np
# import torchmetrics
from time import time
from model import *
from utils import *
from sklearn.metrics import roc_auc_score,average_precision_score,matthews_corrcoef,recall_score,precision_score,f1_score
import pandas as pd
import sys
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from loss import SupConHardLoss
from utils import prepare_tensorboard
from data_clean import prepare_dataloaders as prepare_dataloader_clean 
from data_batchsample import prepare_dataloaders as prepare_dataloader_batchsample
from model import MaskedLMDataCollator

# torch.autograd.set_detect_anomaly(True)

def loss_fix(id_frag, motif_logits, target_frag, tools):
    #id_frag [batch]
    #motif_logits [batch, num_clas, seq]
    #target_frag [batch, num_clas, seq]
    fixed_loss = 0
    for i in range(len(id_frag)):
        frag_ind = id_frag[i].split('@')[1]
        target_thylakoid = target_frag[i, -1]  # -1 for Thylakoid, [seq]; -2 for chloroplast
        # label_first = target_thylakoid[0] # 1 or 0
        target_chlo = target_frag[i, -2]
        if frag_ind == '0' and torch.max(target_chlo) == 0 and torch.max(target_thylakoid) == 1:
            # print("case2")
            l = torch.where(target_thylakoid == 1)[0][0]
            true_chlo = target_frag[i, -2, :(l-1)] == 1
            false_chlo = target_frag[i, -2, :(l-1)] == 0
            motif_logits[i, -2, :(l-1)][true_chlo] = 100
            motif_logits[i, -2, :(l-1)][false_chlo] = -100
    # return fixed_loss
    # return target_frag
    return motif_logits, target_frag



def make_buffer(id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple):
    id_frags_list = []
    seq_frag_list = []
    target_frag_list = []
    for i in range(len(id_frag_list_tuple)):
        id_frags_list.extend(id_frag_list_tuple[i])
        seq_frag_list.extend(seq_frag_list_tuple[i])
        target_frag_list.extend(target_frag_nplist_tuple[i])
    seq_frag_tuple = tuple(seq_frag_list)
    target_frag_pt = torch.from_numpy(np.stack(target_frag_list, axis=0))
    type_protein_pt = torch.stack(list(type_protein_pt_tuple), axis=0)
    return id_frags_list, seq_frag_tuple, target_frag_pt, type_protein_pt


def train_loop(tools, configs, warm_starting, train_writer, epoch):
    # exit(0)
    # if epoch >= configs.train_settings.weighted_loss_sum_start_epoch:
    #     print("loss sum weights: ", configs.train_settings.loss_sum_weights)
    #     customlog(tools["logfilepath"],
    #               "loss sum weights: " + str(configs.train_settings.loss_sum_weights) + '\n')
    # accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=tools['num_classes'], average='macro')
    # f1_score = torchmetrics.F1Score(num_classes=tools['num_classes'], average='macro', task="multiclass")
    # accuracy.to(tools['train_device'])
    # f1_score.to(tools["train_device"])
    global global_step
    tools["optimizer"].zero_grad()
    scaler = GradScaler()
    size = len(tools['train_loader'].dataset)
    num_batches = len(tools['train_loader'])
    print("size="+str(size)+" num_batches="+str(num_batches))
    train_loss = 0
    # cs_num=np.zeros(9)
    # cs_correct=np.zeros(9)
    # type_num=np.zeros(10)
    # type_correct=np.zeros(10)
    # cutoff=0.5
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.train().cuda()
    tools['net'].train().to(tools['train_device'])
    for batch, (id_tuple, id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple, sample_weight_tuple, pos_neg, ORI_AUG) in enumerate(tools['train_loader']):
        if configs.train_settings.ignore_ori:
            print(len(id_tuple), id_tuple)
            print(len(id_frag_list_tuple), id_frag_list_tuple)
            print(len(ORI_AUG), ORI_AUG)
        # exit(0)
        b_size = len(id_tuple)
        flag_batch_extension = False
        if (configs.supcon.apply and not warm_starting and pos_neg is not None) or \
                (configs.supcon.apply and warm_starting):
            
            #customlog(tools["logfilepath"], f"flag_batch_extension")
            #print("flag_batch_extension")
            """
            For two scenarios (CASE B & C, see Encoder::forward()) where the batch needs to be extended,
            extend the 6 tuples with pos_neg
            0 - id_tuple,
            1 - id_frag_list_tuple,
            2 - seq_frag_list_tuple,
            3 - target_frag_nplist_tuple,
            4 - type_protein_pt_tuple,
            5 - and sample_weight_tuple
            without the extending, each len(tuple) == batch_size
            after extending, len(tuple) == batch_size * (1 + n_pos + n_neg)
            """
            flag_batch_extension = True
            for one_in_a_batch in range(b_size):
                # pos_neg[one_in_a_batch][0]
                for one_of_pos in range(configs.supcon.n_pos):
                    # pos_neg[one_in_a_batch][0][one_of_pos]
                    id_tuple += (pos_neg[one_in_a_batch][0][one_of_pos][0],)
                    id_frag_list_tuple += (pos_neg[one_in_a_batch][0][one_of_pos][1],)
                    seq_frag_list_tuple += (pos_neg[one_in_a_batch][0][one_of_pos][2],)
                    target_frag_nplist_tuple += (pos_neg[one_in_a_batch][0][one_of_pos][3],)
                    type_protein_pt_tuple += (pos_neg[one_in_a_batch][0][one_of_pos][4],)
                    sample_weight_tuple += (pos_neg[one_in_a_batch][0][one_of_pos][5],)
            
            for one_in_a_batch in range(b_size):
                # pos_neg[one_in_a_batch][1]
                for one_of_neg in range(configs.supcon.n_neg):
                    # pos_neg[one_in_a_batch][1][one_of_neg]
                    id_tuple += (pos_neg[one_in_a_batch][1][one_of_neg][0],)
                    id_frag_list_tuple += (pos_neg[one_in_a_batch][1][one_of_neg][1],)
                    seq_frag_list_tuple += (pos_neg[one_in_a_batch][1][one_of_neg][2],)
                    target_frag_nplist_tuple += (pos_neg[one_in_a_batch][1][one_of_neg][3],)
                    type_protein_pt_tuple += (pos_neg[one_in_a_batch][1][one_of_neg][4],)
                    sample_weight_tuple += (pos_neg[one_in_a_batch][1][one_of_neg][5],)
            
            type_protein_pt_tuple_temp = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in type_protein_pt_tuple]
            type_protein_pt_tuple = type_protein_pt_tuple_temp
        
        id_frags_list, seq_frag_tuple, target_frag_pt, type_protein_pt = make_buffer(id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple)
        with autocast():
            # Compute prediction and loss
            encoded_seq=tokenize(tools, seq_frag_tuple)

            if type(encoded_seq)==dict:
                for k in encoded_seq.keys():
                    encoded_seq[k]=encoded_seq[k].to(tools['train_device'])
            else:
                encoded_seq=encoded_seq.to(tools['train_device'])

            classification_head, motif_logits, projection_head = tools['net'](
                                 encoded_seq, 
                                 id_tuple, 
                                 id_frags_list, 
                                 seq_frag_tuple, 
                                 pos_neg, 
                                 warm_starting)
            weighted_loss_sum = 0
            weighted_supcon_loss = -1
            class_loss = -1
            position_loss=-1
            if not warm_starting:
                motif_logits, target_frag = loss_fix(id_frags_list, motif_logits, target_frag_pt, tools)
                sample_weight_pt = torch.from_numpy(np.array(sample_weight_tuple)).to(tools['train_device']).unsqueeze(1)

                if configs.train_settings.additional_pos_weights:
                    pass
                    # motif_logits_nucleus = motif_logits[:, 0:1, :]
                    # motif_logits_nucleus_export = motif_logits[:, 4:5, :]
                    # # Concatenate the parts that exclude indices 0 and 4
                    # motif_logits_6 = torch.cat((motif_logits[:, 1:4, :], motif_logits[:, 5:, :]), dim=1)
                    #
                    #
                    # target_frag_nucleus = target_frag[:, 0:1, :]
                    # target_frag_nucleus_export = target_frag[:, 4:5, :]
                    # # Concatenate the parts that exclude indices 0 and 4
                    # target_frag_6 = torch.cat((target_frag[:, 1:4, :], target_frag[:, 5:, :]), dim=1)
                    #
                    # position_loss = tools['loss_function_6'](motif_logits_6, target_frag_6.to(tools['train_device']))
                    # nucleus_position_loss = tools['loss_function_nucleus']\
                    #     (motif_logits_nucleus, target_frag_nucleus.to(tools['train_device']))
                    # nucleus_export_position_loss = tools['loss_function_nucleus_export']\
                    #     (motif_logits_nucleus_export, target_frag_nucleus_export.to(tools['train_device']))
                    #
                    # """
                    # add_position_loss & add_sample_weight_to_position_loss
                    # """
                    # if configs.train_settings.add_sample_weight_to_position_loss:
                    #     print('add_position_loss & add_sample_weight_to_position_loss cannot be True at the sametime')
                    #     customlog(tools["logfilepath"], f'add_position_loss & add_sample_weight_to_position_loss cannot be True at the sametime\n')
                    #     raise 'add_position_loss & add_sample_weight_to_position_loss cannot be True at the sametime'

                else:

                    if 1==1:
                        print(motif_logits.shape)
                        # exit(0)

                    position_loss = torch.mean(tools['loss_function'](motif_logits, target_frag.to(tools['train_device'])))
                    # yichuan: 因为BCEloss已经修改成了none, 所以这里必须要加mean

                    if configs.train_settings.add_sample_weight_to_position_loss:
                        # print(len(id_frags_list), len(sample_weight_tuple))
                        # print(id_frags_list)
                        # print(sample_weight_tuple)
                        new_id_frags_list = [id_frag for id_frag in id_frags_list if id_frag.endswith('@0')]
                        id_to_weight = {id_frag.split('@')[0]: weight for id_frag, weight in
                                        zip(new_id_frags_list, sample_weight_tuple)}
                        # print(id_to_weight)
                        new_sample_weight_list = [id_to_weight.get(id_frag.split('@')[0], 0.0) for id_frag in
                                                  id_frags_list]
                        new_sample_weight_tuple = tuple(new_sample_weight_list)
                        # print(len(new_sample_weight_tuple))
                        # print(new_sample_weight_tuple)
                        new_sample_weight_pt = torch.from_numpy(np.array(new_sample_weight_tuple)).to(tools['train_device']).unsqueeze(1)
                        # print(sample_weight_pt)
                        # print(new_sample_weight_pt)
                        # print(motif_logits.shape)
                        # exit(0)
                        position_loss = torch.mean(tools['loss_function'](motif_logits, target_frag.to(tools['train_device'])) * new_sample_weight_pt.unsqueeze(1))

                
                if configs.train_settings.data_aug.enable:
                    # class_loss = torch.mean(tools['loss_function_pro'](classification_head, type_protein_pt.to(tools['train_device']))) #remove sample_weight_pt
                    # class_loss = torch.mean(tools['loss_function_pro'](classification_head, type_protein_pt.to(tools['train_device'])) * sample_weight_pt)  # - yichuan 0526
                    if configs.train_settings.add_sample_weight_to_class_loss_when_data_aug:
                        class_loss = torch.mean(tools['loss_function_pro'](classification_head, type_protein_pt.to(
                            tools['valid_device'])) * sample_weight_pt)  # yichuan
                    else:
                        class_loss = torch.mean(
                            tools['loss_function_pro'](classification_head, type_protein_pt.to(tools['valid_device'])))
                else:
                    class_loss = torch.mean(tools['loss_function_pro'](classification_head, type_protein_pt.to(tools['train_device'])) * sample_weight_pt)

                if configs.train_settings.additional_pos_weights:
                    pass
                    # train_writer.add_scalar('step class_loss', class_loss.item(), global_step=global_step)
                    # train_writer.add_scalar('step position_loss_6', position_loss.item(), global_step=global_step)
                    # train_writer.add_scalar('step nucleus_position_loss', nucleus_position_loss.item(),
                    #                         global_step=global_step)
                    # train_writer.add_scalar('step nucleus_export_position_loss', nucleus_export_position_loss.item(),
                    #                         global_step=global_step)
                    #
                    # element_num = motif_logits.numel()
                    # position_loss_sum = (position_loss + nucleus_position_loss + nucleus_export_position_loss)/element_num
                    # # print(element_num)
                    # # exit(0)
                    #
                    # print(f"{global_step} class_loss:{class_loss.item()}  position_loss:{position_loss_sum}  " +
                    #       f"element_num:{element_num}  " +
                    #       f"position_loss_6(sum):{position_loss.item()}  " +
                    #       f"nucleus_position_loss(sum):{nucleus_position_loss.item()}  " +
                    #       f"nucleus_export_position_loss(sum):{nucleus_export_position_loss.item()}")
                    # # weighted_loss_sum = class_loss + position_loss_sum
                    # # if epoch >= configs.train_settings.weighted_loss_sum_start_epoch:  # yichuan 0529
                    # #     weighted_loss_sum = class_loss * configs.train_settings.loss_sum_weights[0] + position_loss_sum * configs.train_settings.loss_sum_weights[1]
                    # # else:
                    # # Simplified code for weighted loss calculation
                    # # position_loss_weighted = position_loss / configs.train_settings.position_loss_T if configs.train_settings.position_loss_T != 1 else position_loss
                    # # if configs.train_settings.only_use_position_loss:
                    # #     weighted_loss_sum = position_loss_weighted  # yichuan updated on 0610 and 0601
                    # # else:
                    # #     weighted_loss_sum = class_loss + position_loss_weighted  # yichuan updated on 0612
                    #
                    # # Determine the weighted position loss based on a configurable threshold.
                    # if configs.train_settings.position_loss_T != 1:
                    #     position_loss_weighted = position_loss / configs.train_settings.position_loss_T
                    # else:
                    #     position_loss_weighted = position_loss
                    #
                    # # Determine the weighted class loss based on a configurable threshold.
                    # if configs.train_settings.class_loss_T != 1:
                    #     class_loss_weighted = class_loss / configs.train_settings.class_loss_T
                    # else:
                    #     class_loss_weighted = class_loss
                    #
                    # # Calculate the weighted sum of losses.
                    # if configs.train_settings.only_use_position_loss:
                    #     # If configured to only use position loss, ignore class loss.
                    #     weighted_loss_sum = position_loss_weighted
                    # else:
                    #     # Otherwise, sum class loss and weighted position loss.
                    #     weighted_loss_sum = class_loss_weighted + position_loss_weighted



                else:
                    train_writer.add_scalar('step class_loss', class_loss.item(), global_step=global_step)
                    # train_writer.add_scalar('step position_loss', position_loss.item(), global_step=global_step)
                    print(f"{global_step} class_loss:{class_loss.item()}  position_loss:{position_loss.item()}")
                    # weighted_loss_sum = class_loss + position_loss
                    # if epoch >= configs.train_settings.weighted_loss_sum_start_epoch:  # yichuan 0529
                    #     weighted_loss_sum = class_loss * configs.train_settings.loss_sum_weights[0] + position_loss * configs.train_settings.loss_sum_weights[1]
                    # else:
                    # Simplified code for weighted loss calculation
                    # position_loss_weighted = position_loss / configs.train_settings.position_loss_T if configs.train_settings.position_loss_T != 1 else position_loss
                    # if configs.train_settings.only_use_position_loss:
                    #     weighted_loss_sum = position_loss_weighted  # yichuan updated on 0610 and 0601
                    # else:
                    #     weighted_loss_sum = class_loss + position_loss_weighted  # yichuan updated on 0612

                    # Determine the weighted position loss based on a configurable threshold.
                    if configs.train_settings.position_loss_T != 1:
                        position_loss_weighted = position_loss / configs.train_settings.position_loss_T
                    else:
                        position_loss_weighted = position_loss

                    # Determine the weighted class loss based on a configurable threshold.
                    if configs.train_settings.class_loss_T != 1:
                        class_loss_weighted = class_loss / configs.train_settings.class_loss_T
                    else:
                        class_loss_weighted = class_loss

                    # Calculate the weighted sum of losses.
                    if configs.train_settings.only_use_position_loss:
                        # If configured to only use position loss, ignore class loss.
                        weighted_loss_sum = position_loss_weighted
                    else:
                        # Otherwise, sum class loss and weighted position loss.
                        weighted_loss_sum = class_loss_weighted + position_loss_weighted

            if configs.supcon.apply and configs.supcon.apply_supcon_loss: #configs.supcon.apply: # and warm_starting: calculate supcon loss no matter whether warm_starting or not.
                supcon_loss = tools['loss_function_supcon'](
                                projection_head,
                                configs.supcon.temperature,
                                configs.supcon.n_pos)
                weighted_supcon_loss = configs.supcon.weight * supcon_loss
                print(f"{global_step} supcon_loss:{weighted_supcon_loss.item()}")
                train_writer.add_scalar('step supcon_loss', weighted_supcon_loss.item(), global_step=global_step)
                weighted_loss_sum += weighted_supcon_loss
            if configs.supcon.apply is False and warm_starting:
                raise ValueError("Check configs.supcon.apply and configs.supcon.warm_start")

            train_loss += weighted_loss_sum.item()
        
        # Backpropagation
        scaler.scale(weighted_loss_sum).backward()
        scaler.step(tools['optimizer'])
        scaler.update()
        tools['scheduler'].step()
        print(f"{global_step} loss:{weighted_loss_sum.item()}\n")
        train_writer.add_scalar('step loss', weighted_loss_sum.item(), global_step=global_step)
        train_writer.add_scalar('learning_rate', tools['scheduler'].get_lr()[0], global_step=global_step)
        #if global_step % configs.train_settings.log_every == 0: #30 before changed into 0
        if batch % configs.train_settings.log_every == 0: #for comparison with original codes
            loss, current = weighted_loss_sum.item(), (batch + 1) * b_size  # len(id_tuple)
            if flag_batch_extension:
                customlog(tools["logfilepath"], f"{global_step} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  ->  " +
                      f"[{(batch + 1) * len(id_tuple):>5d}/{size*(1+configs.supcon.n_pos+configs.supcon.n_neg):>5d}]")
            else:
                customlog(tools["logfilepath"], f"{global_step} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")
            if class_loss !=-1:
                if configs.train_settings.additional_pos_weights:  # yichuan 0529
                    # customlog(tools["logfilepath"],
                    #           f"{global_step} class loss: {class_loss.item():>7f} position_loss:{position_loss_sum.item():>7f}\n")
                    pass
                else:
                    customlog(tools["logfilepath"], f"{global_step} class loss: {class_loss.item():>7f} position_loss:{position_loss.item():>7f}\n")

            if weighted_supcon_loss !=-1:
               customlog(tools["logfilepath"], f"{global_step} supcon loss: {weighted_supcon_loss.item():>7f}\n")
        
        global_step+=1
    
    epoch_loss = train_loss/num_batches
    return epoch_loss



def test_loop(tools, dataloader,train_writer,valid_writer,configs):
    customlog(tools["logfilepath"], f'number of test steps per epoch: {len(dataloader)}\n')
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.eval().cuda()
    tools['net'].eval().to(tools["valid_device"])
    num_batches = len(dataloader)
    test_loss=0
    test_class_loss=0
    test_position_loss=0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    #print("in test loop")
    with torch.no_grad():
        for batch, (id_tuple, id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple, sample_weight_tuple, pos_neg) in enumerate(dataloader):
            id_frags_list, seq_frag_tuple, target_frag_pt, type_protein_pt = make_buffer(id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple)
            encoded_seq=tokenize(tools, seq_frag_tuple)
            if type(encoded_seq)==dict:
                for k in encoded_seq.keys():
                    encoded_seq[k]=encoded_seq[k].to(tools['valid_device'])
            else:
                encoded_seq=encoded_seq.to(tools['valid_device'])
            #print("ok1")
            classification_head, motif_logits, projection_head = tools['net'](
                       encoded_seq,
                       id_tuple,id_frags_list,seq_frag_tuple, 
                       None, False) #for test_loop always used None and False!
            #print("ok2")
            weighted_loss_sum = 0
            #if not warm_starting:
            class_loss = 0
            position_loss=0
            motif_logits, target_frag = loss_fix(id_frags_list, motif_logits, target_frag_pt, tools)
            sample_weight_pt = torch.from_numpy(np.array(sample_weight_tuple)).to(tools['valid_device']).unsqueeze(1)
            # position_loss = tools['loss_function'](motif_logits, target_frag.to(tools['valid_device']))
            position_loss = torch.mean(tools['loss_function'](motif_logits, target_frag.to(tools['valid_device'])))
            # yichuan: 因为BCEloss已经修改成了none, 所以这里必须要加mean
            if configs.train_settings.add_sample_weight_to_position_loss:
                # print(len(id_frags_list), len(sample_weight_tuple))
                # print(id_frags_list)
                # print(sample_weight_tuple)
                new_id_frags_list = [id_frag for id_frag in id_frags_list if id_frag.endswith('@0')]
                id_to_weight = {id_frag.split('@')[0]: weight for id_frag, weight in
                                zip(new_id_frags_list, sample_weight_tuple)}
                # print(id_to_weight)
                new_sample_weight_list = [id_to_weight.get(id_frag.split('@')[0], 0.0) for id_frag in
                                          id_frags_list]
                new_sample_weight_tuple = tuple(new_sample_weight_list)
                # print(len(new_sample_weight_tuple))
                # print(new_sample_weight_tuple)
                new_sample_weight_pt = torch.from_numpy(np.array(new_sample_weight_tuple)).to(
                    tools['valid_device']).unsqueeze(1)
                # print(new_sample_weight_pt.shape)
                # print(new_sample_weight_pt)
                position_loss = torch.mean(tools['loss_function'](motif_logits, target_frag.to(
                    tools['valid_device'])) * new_sample_weight_pt.unsqueeze(1))

            if configs.train_settings.data_aug.enable:
                # class_loss = torch.mean(tools['loss_function_pro'](classification_head, type_protein_pt.to(tools['valid_device'])))
                if configs.train_settings.add_sample_weight_to_class_loss_when_data_aug:
                    class_loss = torch.mean(tools['loss_function_pro'](classification_head, type_protein_pt.to(tools['valid_device'])) * sample_weight_pt)  # yichuan
                else:
                    class_loss = torch.mean(tools['loss_function_pro'](classification_head, type_protein_pt.to(tools['valid_device'])))
            else:
                class_loss = torch.mean(tools['loss_function_pro'](classification_head, type_protein_pt.to(tools['valid_device'])) * sample_weight_pt)

            # Simplified code for weighted loss calculation
            # position_loss_weighted = position_loss / configs.train_settings.position_loss_T if configs.train_settings.position_loss_T != 1 else position_loss
            # if configs.train_settings.only_use_position_loss:
            #     weighted_loss_sum = position_loss_weighted  # yichuan updated on 0610 and 0601
            # else:
            #     weighted_loss_sum = class_loss + position_loss_weighted  # yichuan updated on 0612
            # Determine the weighted position loss based on a configurable threshold.
            if configs.train_settings.position_loss_T != 1:
                position_loss_weighted = position_loss / configs.train_settings.position_loss_T
            else:
                position_loss_weighted = position_loss

            # Determine the weighted class loss based on a configurable threshold.
            if configs.train_settings.class_loss_T != 1:
                class_loss_weighted = class_loss / configs.train_settings.class_loss_T
            else:
                class_loss_weighted = class_loss

            # Calculate the weighted sum of losses.
            if configs.train_settings.only_use_position_loss:
                # If configured to only use position loss, ignore class loss.
                weighted_loss_sum = position_loss_weighted
            else:
                # Otherwise, sum class loss and weighted position loss.
                weighted_loss_sum = class_loss_weighted + position_loss_weighted

            """
            if configs.supcon.apply and warm_starting:
                supcon_loss = tools['loss_function_supcon'](
                                    projection_head,
                                    configs.supcon.temperature,
                                    configs.supcon.n_pos)
                weighted_loss_sum += configs.supcon.weight * supcon_loss
            """
            test_loss += weighted_loss_sum.item()
            test_position_loss += position_loss.item()
            test_class_loss += class_loss.item()
            # label = torch.argmax(label_1hot, dim=1)
            # type_pred = torch.argmax(type_probab, dim=1)
            # accuracy.update(type_pred.detach(), label.detach().to(tools['valid_device']))
            # macro_f1_score.update(type_pred.detach(), label.detach().to(tools['valid_device']))
            # f1_score.update(type_pred.detach(), label.detach().to(tools['valid_device']))

        test_loss = test_loss / num_batches
        test_position_loss = test_position_loss / num_batches
        test_class_loss = test_class_loss / num_batches
        # epoch_acc = np.array(accuracy.compute().cpu())
        # epoch_macro_f1 = macro_f1_score.compute().cpu().item()
        # epoch_f1 = np.array(f1_score.compute().cpu())
        # acc_cs = cs_correct / cs_num
    return test_loss,test_class_loss,test_position_loss


def frag2protein(data_dict, tools):
    overlap=tools['frag_overlap']
    # no_overlap=tools['max_len']-2-overlap
    for id_protein in data_dict.keys():
        id_frag_list = data_dict[id_protein]['id_frag']
        seq_protein=""
        motif_logits_protein=np.array([])
        motif_target_protein=np.array([])
        for i in range(len(id_frag_list)):
            id_frag = id_protein+"@"+str(i)
            ind = id_frag_list.index(id_frag)
            seq_frag = data_dict[id_protein]['seq_frag'][ind]
            target_frag = data_dict[id_protein]['target_frag'][ind]
            motif_logits_frag = data_dict[id_protein]['motif_logits'][ind]
            l=len(seq_frag)
            if i==0:
                seq_protein=seq_frag
                motif_logits_protein=motif_logits_frag[:,:l]
                motif_target_protein=target_frag[:,:l]
            else:
                seq_protein = seq_protein + seq_frag[overlap:]
                # x_overlap = np.maximum(motif_logits_protein[:,-overlap:], motif_logits_frag[:,:overlap])
                x_overlap = (motif_logits_protein[:,-overlap:] + motif_logits_frag[:,:overlap])/2
                motif_logits_protein = np.concatenate((motif_logits_protein[:,:-overlap], x_overlap, motif_logits_frag[:,overlap:l]),axis=1)
                motif_target_protein = np.concatenate((motif_target_protein, target_frag[:,overlap:l]), axis=1)
        data_dict[id_protein]['seq_protein']=seq_protein
        data_dict[id_protein]['motif_logits_protein']=motif_logits_protein
        data_dict[id_protein]['motif_target_protein']=motif_target_protein
    return data_dict


def get_data_dict(args, dataloader, tools):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.eval().cuda()
    model_path = os.path.join(tools['checkpoint_path'], f'best_model.pth')
    if args.predict == 1 and os.path.exists(args.resume_path):
        model_path = args.resume_path

    print('model_path:', model_path)
    model_checkpoint = torch.load(model_path, map_location='cpu')
    tools['net'].load_state_dict(model_checkpoint['model_state_dict'])
    tools['net'].eval().to(tools["valid_device"])
    n = tools['num_classes']

    # cutoff = tools['cutoff']
    data_dict = {}
    with torch.no_grad():
        # for batch, (id, id_frags, seq_frag, target_frag, type_protein) in enumerate(dataloader):
        for batch, (id_tuple, id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple, sample_weight_tuple, pos_neg) in enumerate(dataloader):
            # id_frags_list, seq_frag_tuple, target_frag_tuple = make_buffer(id_frags, seq_frag, target_frag)
            id_frags_list, seq_frag_tuple, target_frag_pt, type_protein_pt = make_buffer(id_frag_list_tuple,
                                                                                         seq_frag_list_tuple,
                                                                                         target_frag_nplist_tuple,
                                                                                         type_protein_pt_tuple)
            encoded_seq = tokenize(tools, seq_frag_tuple)
            if type(encoded_seq) == dict:
                for k in encoded_seq.keys():
                    encoded_seq[k] = encoded_seq[k].to(tools['valid_device'])
            else:
                encoded_seq = encoded_seq.to(tools['valid_device'])
            classification_head, motif_logits, whatever = tools['net'](encoded_seq, id_tuple, id_frags_list, seq_frag_tuple, None, False)
            m = torch.nn.Sigmoid()
            motif_logits = m(motif_logits)
            classification_head = m(classification_head)

            x_frag = np.array(motif_logits.cpu())  # [batch, head, seq]
            y_frag = np.array(target_frag_pt.cpu())  # [batch, head, seq]
            x_pro = np.array(classification_head.cpu())  # [sample, n]
            y_pro = np.array(type_protein_pt.cpu())  # [sample, n]
            for i in range(len(id_frags_list)):
                id_protein = id_frags_list[i].split('@')[0]
                j = id_tuple.index(id_protein)
                if id_protein in data_dict.keys():
                    data_dict[id_protein]['id_frag'].append(id_frags_list[i])
                    data_dict[id_protein]['seq_frag'].append(seq_frag_tuple[i])
                    data_dict[id_protein]['target_frag'].append(y_frag[i])  # [[head, seq], ...]
                    data_dict[id_protein]['motif_logits'].append(x_frag[i])  # [[head, seq], ...]
                else:
                    data_dict[id_protein] = {}
                    data_dict[id_protein]['id_frag'] = [id_frags_list[i]]
                    data_dict[id_protein]['seq_frag'] = [seq_frag_tuple[i]]
                    data_dict[id_protein]['target_frag'] = [y_frag[i]]
                    data_dict[id_protein]['motif_logits'] = [x_frag[i]]
                    data_dict[id_protein]['type_pred'] = x_pro[j]
                    data_dict[id_protein]['type_target'] = y_pro[j]

        data_dict = frag2protein(data_dict, tools)
    return data_dict


def evaluate_protein(data_dict, tools, constrain):
    n = tools['num_classes']
    # IoU_difcut=np.zeros([n, 9])
    # FDR_frag_difcut=np.zeros([1,9])
    IoU_pro_difcut = np.zeros([n, 9, 9])  # just for nuc and nuc_export
    # FDR_pro_difcut=np.zeros([1,9])
    result_pro_difcut = np.zeros([n, 6, 9])
    cs_acc_difcut = np.zeros([n, 9])
    classname = ["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
                 "SIGNAL", "chloroplast", "Thylakoid"]
    criteria = ["roc_auc_score", "average_precision_score", "matthews_corrcoef",
                "recall_score", "precision_score", "f1_score"]
    cutoffs = [x / 10 for x in range(1, 10)]
    cutoff_dim_pro = 0
    for cutoff_pro in cutoffs:
        cutoff_dim_aa = 0
        for cutoff_aa in cutoffs:
            cutoff_pro_list = [cutoff_pro] * n
            cutoff_aa_list = [cutoff_aa] * n
            scores = get_scores(tools, cutoff_pro_list, cutoff_aa_list, n, data_dict, constrain)
            # IoU_difcut[:,cut_dim]=scores['IoU']
            # IoU_difcut[:,cut_dim]=np.array([float("{:.3f}".format(i)) for i in scores['IoU']])
            # FDR_frag_difcut[:,cut_dim]=scores['FDR_frag']
            # FDR_frag_difcut[:,cut_dim]=float("{:.3f}".format(scores['FDR_frag']))
            IoU_pro_difcut[:, cutoff_dim_pro, cutoff_dim_aa] = scores['IoU_pro']
            # IoU_pro_difcut[:,cut_dim]=np.array([float("{:.3f}".format(i)) for i in scores['IoU_pro']])
            # FDR_pro_difcut[:,cut_dim]=scores['FDR_pro']
            # FDR_pro_difcut[:,cut_dim]=float("{:.3f}".format(scores['FDR_pro']))
            # result_pro_difcut[:,:,cut_dim]=np.array([float("{:.3f}".format(i)) for i in scores['result_pro'].reshape(-1)]).reshape(scores['result_pro'].shape)
            cs_acc_difcut[:, cutoff_dim_pro] = scores['cs_acc']
            cutoff_dim_aa += 1
        result_pro_difcut[:, :, cutoff_dim_pro] = scores['result_pro']
        cutoff_dim_pro += 1

    # print(cutoffs)
    opti_cutoffs_pro = [0] * n
    opti_cutoffs_aa = [0] * n
    for head in range(n):
        best_f1 = -np.inf
        best_IoU = -np.inf
        best_cs = -np.inf
        for index in range(len(cutoffs)):
            f1 = result_pro_difcut[head, -1, index]  # -1 is f1 score
            if f1 > best_f1:
                best_f1 = f1
                opti_cutoffs_pro[head] = cutoffs[index]
            if head == 0 or head == 4:
                if constrain:
                    index_pro = cutoffs.index(opti_cutoffs_pro[head])
                    IoU = IoU_pro_difcut[
                        head, index_pro, index]  # dim 2 could be any when constrain is false, lv1 and lv2 are independent
                else:
                    IoU = IoU_pro_difcut[
                        head, 0, index]  # dim 2 could be any when constrain is false, lv1 and lv2 are independent
                if IoU > best_IoU:
                    best_IoU = IoU
                    opti_cutoffs_aa[head] = cutoffs[index]
            else:
                cs = cs_acc_difcut[head, index]
                if cs > best_cs:
                    best_cs = cs
                    opti_cutoffs_aa[head] = cutoffs[index]

    IoU_pro_difcut = np.zeros([n])  # just for nuc and nuc_export
    # result_pro_difcut=np.zeros([n,6,9])
    result_pro_difcut = np.zeros([n, 6])
    # cs_acc_difcut=np.zeros([n, 9])
    cs_acc_difcut = np.zeros([n])
    scores = get_scores(tools, opti_cutoffs_pro, opti_cutoffs_aa, n, data_dict, constrain)
    IoU_pro_difcut = scores['IoU_pro']
    TPR_FPR_FNR_difcut = scores['TPR_FPR_FNR']
    cs_acc_difcut = scores['cs_acc']
    result_pro_difcut = scores['result_pro']

    customlog(tools["logfilepath"], f"===================Evaluate protein results========================\n")
    customlog(tools["logfilepath"], f" optimized cutoffs: \n")
    cutoffs = np.array([opti_cutoffs_pro, opti_cutoffs_aa])
    cutoffs = pd.DataFrame(cutoffs, columns=classname, index=["cutoffs_pro", "cutoffs_aa"])
    customlog(tools["logfilepath"], cutoffs.__repr__())

    customlog(tools["logfilepath"], f"===========================================\n")
    customlog(tools["logfilepath"], f" Jaccard Index (protein): \n")
    IoU_pro_difcut = pd.DataFrame(IoU_pro_difcut, index=classname)
    IoU_pro_difcut_selected_rows = IoU_pro_difcut.iloc[[0, 4]]
    customlog(tools["logfilepath"], IoU_pro_difcut_selected_rows.__repr__())

    customlog(tools["logfilepath"], f"===========================================\n")
    customlog(tools["logfilepath"], f" TPR, FPR, FNR: \n")
    TPR_FPR_FNR_difcut = pd.DataFrame(TPR_FPR_FNR_difcut, index=classname)
    TPR_FPR_FNR_difcut_selected_rows = TPR_FPR_FNR_difcut.iloc[[0, 4]]
    customlog(tools["logfilepath"], TPR_FPR_FNR_difcut_selected_rows.__repr__())

    customlog(tools["logfilepath"], f"===========================================\n")
    customlog(tools["logfilepath"], f" cs acc: \n")
    cs_acc_difcut = pd.DataFrame(cs_acc_difcut, index=classname)
    rows_to_exclude = [0, 4]
    filtered_df = cs_acc_difcut.drop(cs_acc_difcut.index[rows_to_exclude])
    customlog(tools["logfilepath"], filtered_df.__repr__())

    customlog(tools["logfilepath"], f"===========================================\n")
    customlog(tools["logfilepath"], f" Class prediction performance: \n")
    tem = pd.DataFrame(result_pro_difcut, columns=criteria, index=classname)
    customlog(tools["logfilepath"], tem.__repr__())

    return opti_cutoffs_pro, opti_cutoffs_aa


def test_protein(data_dict, tools, opti_cutoffs_pro, opti_cutoffs_aa, constrain):
    n = tools['num_classes']

    classname = ["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
                 "SIGNAL", "chloroplast", "Thylakoid"]
    criteria = ["roc_auc_score", "average_precision_score", "matthews_corrcoef",
                "recall_score", "precision_score", "f1_score"]

    IoU_pro_difcut = np.zeros([n])  # just for nuc and nuc_export
    # result_pro_difcut=np.zeros([n,6,9])
    result_pro_difcut = np.zeros([n, 6])
    # cs_acc_difcut=np.zeros([n, 9])
    cs_acc_difcut = np.zeros([n])
    scores = get_scores(tools, opti_cutoffs_pro, opti_cutoffs_aa, n, data_dict, constrain)
    IoU_pro_difcut = scores['IoU_pro']
    TPR_FPR_FNR_difcut = scores['TPR_FPR_FNR']
    cs_acc_difcut = scores['cs_acc']
    result_pro_difcut = scores['result_pro']

    customlog(tools["logfilepath"],
              f"===================Test protein results constrain: {constrain}========================\n")
    customlog(tools["logfilepath"], f" optimized cutoffs: \n")
    cutoffs = np.array([opti_cutoffs_pro, opti_cutoffs_aa])
    cutoffs = pd.DataFrame(cutoffs, columns=classname, index=["cutoffs_pro", "cutoffs_aa"])
    customlog(tools["logfilepath"], cutoffs.__repr__())

    customlog(tools["logfilepath"], f"===========================================\n")
    customlog(tools["logfilepath"], f" Jaccard Index (protein): \n")
    IoU_pro_difcut = pd.DataFrame(IoU_pro_difcut, index=classname)
    IoU_pro_difcut_selected_rows = IoU_pro_difcut.iloc[[0, 4]]
    customlog(tools["logfilepath"], IoU_pro_difcut_selected_rows.__repr__())

    customlog(tools["logfilepath"], f"===========================================\n")
    customlog(tools["logfilepath"], f" TPR, FPR, FNR: \n")
    TPR_FPR_FNR_difcut = pd.DataFrame(TPR_FPR_FNR_difcut, index=classname)
    TPR_FPR_FNR_difcut_selected_rows = TPR_FPR_FNR_difcut.iloc[[0, 4]]
    customlog(tools["logfilepath"], TPR_FPR_FNR_difcut_selected_rows.__repr__())

    customlog(tools["logfilepath"], f"===========================================\n")
    customlog(tools["logfilepath"], f" cs acc: \n")
    cs_acc_difcut = pd.DataFrame(cs_acc_difcut, index=classname)
    rows_to_exclude = [0, 4]
    filtered_df = cs_acc_difcut.drop(cs_acc_difcut.index[rows_to_exclude])
    customlog(tools["logfilepath"], filtered_df.__repr__())

    customlog(tools["logfilepath"], f"===========================================\n")
    customlog(tools["logfilepath"], f" Class prediction performance: \n")
    tem = pd.DataFrame(result_pro_difcut, columns=criteria, index=classname)
    customlog(tools["logfilepath"], tem.__repr__())


def get_scores(tools, cutoff_pro, cutoff_aa, n, data_dict, constrain):
    cs_num = np.zeros(n)
    cs_correct = np.zeros(n)
    cs_acc = np.zeros(n)

    # TP_frag=np.zeros(n)
    # FP_frag=np.zeros(n)
    # FN_frag=np.zeros(n)
    # #Intersection over Union (IoU) or Jaccard Index
    # IoU = np.zeros(n)
    # Negtive_detect_num=0
    # Negtive_num=0

    # Yichuan 0612
    TPR_pro_avg = np.zeros(n)
    FPR_pro_avg = np.zeros(n)
    FNR_pro_avg = np.zeros(n)
    TPR_FPR_FNR_pro_avg = [None] * n

    IoU_pro = np.zeros(n)

    # Negtive_detect_pro=0
    # Negtive_pro=0
    result_pro = np.zeros([n, 6])
    for head in range(n):
        x_list = []
        y_list = []
        for id_protein in data_dict.keys():
            x_pro = data_dict[id_protein]['type_pred'][head]  # [1]
            y_pro = data_dict[id_protein]['type_target'][head]  # [1]
            x_list.append(x_pro)
            y_list.append(y_pro)
            if constrain:
                condition = x_pro >= cutoff_pro[head]
            else:
                condition = True
            if y_pro == 1 and condition:
                x_frag = data_dict[id_protein]['motif_logits_protein'][head]  # [seq]
                y_frag = data_dict[id_protein]['motif_target_protein'][head]
                # Negtive_pro += np.sum(np.max(y)==0)
                # Negtive_detect_pro += np.sum((np.max(y)==0) * (np.max(x>=cutoff)==1))
                TPR_pro = np.sum((x_frag >= cutoff_aa[head]) * (y_frag == 1)) / np.sum(y_frag == 1)
                FPR_pro = np.sum((x_frag >= cutoff_aa[head]) * (y_frag == 0)) / np.sum(y_frag == 0)
                FNR_pro = np.sum((x_frag < cutoff_aa[head]) * (y_frag == 1)) / np.sum(y_frag == 1)
                # x_list.append(np.max(x))
                # y_list.append(np.max(y))
                IoU_pro[head] += TPR_pro / (TPR_pro + FPR_pro + FNR_pro)
                TPR_pro_avg[head] += TPR_pro
                FPR_pro_avg[head] += FPR_pro
                FNR_pro_avg[head] += FNR_pro

                cs_num[head] += np.sum(y_frag == 1) > 0
                if np.sum(y_frag == 1) > 0:
                    cs_correct[head] += (np.argmax(x_frag) == np.argmax(y_frag))

        # IoU_pro[head] = TPR_pro[head] / (TPR_pro[head] + FPR_pro[head] + FNR_pro[head])
        TPR_pro_avg[head] = TPR_pro_avg[head] / sum(y_list)
        FPR_pro_avg[head] = FPR_pro_avg[head] / sum(y_list)
        FNR_pro_avg[head] = FNR_pro_avg[head] / sum(y_list)
        TPR_FPR_FNR_pro_avg[head] = (TPR_pro_avg[head], FPR_pro_avg[head], FNR_pro_avg[head])

        IoU_pro[head] = IoU_pro[head] / sum(y_list)
        cs_acc[head] = cs_correct[head] / cs_num[head]

        pred = np.array(x_list)
        target = np.array(y_list)
        # print(target)
        # print(pred)
        # print(cutoff_pro[head])

        # result_pro[head, 0] = roc_auc_score(target, pred)
        # result_pro[head, 1] = average_precision_score(target, pred)
        # result_pro[head, 2] = matthews_corrcoef(target, pred >= cutoff_pro[head])
        # result_pro[head, 3] = recall_score(target, pred >= cutoff_pro[head])
        # result_pro[head, 4] = precision_score(target, pred >= cutoff_pro[head])
        # result_pro[head, 5] = f1_score(target, pred >= cutoff_pro[head])

        try:
            result_pro[head, 0] = roc_auc_score(target, pred)
        except ValueError:
            result_pro[head, 0] = np.nan
        try:
            result_pro[head, 1] = average_precision_score(target, pred)
        except ValueError:
            result_pro[head, 1] = np.nan
        try:
            result_pro[head, 2] = matthews_corrcoef(target, pred >= cutoff_pro[head])
        except ValueError:
            result_pro[head, 2] = np.nan
        try:
            result_pro[head, 3] = recall_score(target, pred >= cutoff_pro[head])
        except ValueError:
            result_pro[head, 3] = np.nan
        try:
            result_pro[head, 4] = precision_score(target, pred >= cutoff_pro[head])
        except ValueError:
            result_pro[head, 4] = np.nan
        try:
            result_pro[head, 5] = f1_score(target, pred >= cutoff_pro[head])
        except ValueError:
            result_pro[head, 5] = np.nan

    scores = {"IoU_pro": IoU_pro,  # [n]
              "result_pro": result_pro,  # [n, 6]
              "cs_acc": cs_acc,  # [n]
              "TPR_FPR_FNR": TPR_FPR_FNR_pro_avg}
    return scores


def main(config_dict, args,valid_batch_number, test_batch_number):
    configs = load_configs(config_dict,args)
    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)
    
    torch.cuda.empty_cache()
    curdir_path, result_path, checkpoint_path, logfilepath = prepare_saving_dir(configs,args.config_path)
    
    train_writer, valid_writer = prepare_tensorboard(result_path)
    npz_file = os.path.join(curdir_path, "targetp_data.npz")
    seq_file = os.path.join(curdir_path, "idmapping_2023_08_25.tsv")
    
    customlog(logfilepath, f'use k-fold index: {valid_batch_number}\n')
    
    if configs.train_settings.dataloader=="batchsample":
        dataloaders_dict = prepare_dataloader_batchsample(configs, valid_batch_number, test_batch_number)
    elif configs.train_settings.dataloader=="clean":
            dataloaders_dict = prepare_dataloader_clean(configs, valid_batch_number, test_batch_number)
    
    #debug_dataloader(dataloaders_dict["train"]) #981
    customlog(logfilepath, "Done Loading data\n")
    customlog(logfilepath, f'number of steps for training data: {len(dataloaders_dict["train"])}\n')
    customlog(logfilepath, f'number of steps for valid data: {len(dataloaders_dict["valid"])}\n')
    customlog(logfilepath, f'number of steps for test data: {len(dataloaders_dict["test"])}\n')
    print(f'number of steps for training data: {len(dataloaders_dict["train"])}\n')
    print(f'number of steps for valid data: {len(dataloaders_dict["valid"])}\n')
    print(f'number of steps for test data: {len(dataloaders_dict["test"])}\n')
    # #debug_dataloader(dataloaders_dict["train"]) #981
    # tokenizer = prepare_tokenizer(configs, curdir_path)
    # customlog(logfilepath, "Done initialize tokenizer\n")
    # #debug_dataloader(dataloaders_dict["train"]) #981
    # if hasattr(configs.train_settings,"MLM") and configs.train_settings.MLM.enable:
    #    masked_lm_data_collator = MaskedLMDataCollator(tokenizer, mlm_probability=configs.train_settings.MLM.mask_ratio)
    # else:
    #    masked_lm_data_collator=None
    #
    # """
    # #for debug
    # config_path = "/data/duolin/MUTargetCLEAN/MUTarget-main/config.yaml"
    # with open(config_path) as file:
    #     config_dict = yaml.full_load(file)
    #
    # configs = load_configs(config_dict)
    # #debug_dataloader(dataloaders_dict["train"]) #981 after prepare_models become 936
    # """
    # encoder = prepare_models(configs, logfilepath, curdir_path)
    # #debug_dataloader(dataloaders_dict["train"]) #936 after prepare_models become 936 , if use same config, 1575!
    # print("Done initialize model\n")
    # customlog(logfilepath, "Done initialize model\n")

    """adapter 0611 begin"""
    if configs.encoder.composition=="official_esm_v2":
        encoder=prepare_models(configs,logfilepath, curdir_path)
        customlog(logfilepath, "Done initialize model\n")
        print("Done initialize model\n")

        alphabet = encoder.model.alphabet
        tokenizer = alphabet.get_batch_converter(
            truncation_seq_length=configs.encoder.max_len
        )
        customlog(logfilepath, "Done initialize tokenizer\n")
        print("Done initialize tokenizer\n")
    else:
        tokenizer=prepare_tokenizer(configs, curdir_path)
        customlog(logfilepath, "Done initialize tokenizer\n")
        print("Done initialize tokenizer\n")

        encoder=prepare_models(configs,logfilepath, curdir_path)
        customlog(logfilepath, "Done initialize model\n")
        print("Done initialize model\n")
    """adapter 0611 end"""

    optimizer, scheduler = prepare_optimizer(encoder, configs, len(dataloaders_dict["train"]), logfilepath)
    if configs.optimizer.mode == 'skip':
        scheduler = optimizer
    customlog(logfilepath, 'preparing optimizer is done\n')
    if args.predict !=1:
       encoder, start_epoch = load_checkpoints(configs, optimizer, scheduler, logfilepath, encoder)
    # print('start epoch', start_epoch)
    # exit(0)
    # w=(torch.ones([9,1,1])*5).to(configs.train_settings.device)
    w = torch.tensor(configs.train_settings.loss_pos_weight, dtype=torch.float32).to(configs.train_settings.device)
    w_nucleus = torch.tensor(configs.train_settings.loss_pos_weight_nucleus, dtype=torch.float32).to(configs.train_settings.device)
    w_nucleus_export = torch.tensor(configs.train_settings.loss_pos_weight_nucleus_export, dtype=torch.float32).to(configs.train_settings.device)
    #debug_dataloader(dataloaders_dict["train"]) #936 after call dataloaders_dict['train']
    
    tools = {
        'frag_overlap': configs.encoder.frag_overlap,
        'cutoffs': configs.predict_settings.cutoffs,
        'composition': configs.encoder.composition, 
        'max_len': configs.encoder.max_len,
        'tokenizer': tokenizer,
        'prm4prmpro': configs.encoder.prm4prmpro,
        'net': encoder,
        'train_loader': dataloaders_dict["train"],
        'valid_loader': dataloaders_dict["valid"],
        'test_loader': dataloaders_dict["test"],
        'train_device': configs.train_settings.device,
        'valid_device': configs.valid_settings.device,
        'train_batch_size': configs.train_settings.batch_size,
        'valid_batch_size': configs.valid_settings.batch_size,
        'optimizer': optimizer,
        # 'loss_function': torch.nn.CrossEntropyLoss(reduction="none"),
        # 'loss_function': torch.nn.BCEWithLogitsLoss(pos_weight=w, reduction="mean"),
        'loss_function': torch.nn.BCEWithLogitsLoss(pos_weight=w, reduction="none"),
        # yichuan: loss_function的BCEWithLogitsLoss原来是mean, 被我改成none, 我在代码中加入了torch.mean
        'loss_function_6': torch.nn.BCEWithLogitsLoss(pos_weight=w, reduction="sum"),
        'loss_function_nucleus': torch.nn.BCEWithLogitsLoss(pos_weight=w_nucleus, reduction="sum"),
        'loss_function_nucleus_export': torch.nn.BCEWithLogitsLoss(pos_weight=w_nucleus_export, reduction="sum"),
        'pos_weight': w,
        #'loss_function': torch.nn.BCELoss(reduction="none"),
        #'loss_function_pro': torch.nn.BCELoss(reduction="none"),
        'loss_function_pro': torch.nn.BCEWithLogitsLoss(reduction="none"),
        'loss_function_supcon': SupConHardLoss,  # Yichuan
        'checkpoints_every': configs.checkpoints_every,
        'scheduler': scheduler,
        'result_path': result_path,
        'checkpoint_path': checkpoint_path,
        'logfilepath': logfilepath,
        'num_classes': configs.encoder.num_classes,
        # 'masked_lm_data_collator': masked_lm_data_collator,
    }
    if args.predict !=1:
        customlog(logfilepath, f'number of train steps per epoch: {len(tools["train_loader"])}\n')
        customlog(logfilepath, "Start training...\n")
        
        best_valid_loss = np.inf
        best_valid_position_loss = np.inf
        global global_step
        global_step=0
        if configs.train_settings.dataloader=="clean":
           total_steps_per_epoch = int(len(tools["train_loader"].dataset.samples)/configs.train_settings.batch_size)
        
        for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):
            warm_starting = False
            if epoch < configs.supcon.warm_start:
                warm_starting = True
                if epoch ==0:
                   print('== Warm Start Began    ==')
                   customlog(logfilepath,f"== Warm Start Began ==\n")
        
        
            if epoch == configs.supcon.warm_start:
                best_valid_loss = np.inf #reset best_valid_loss when warmend ends
                best_valid_position_loss = np.inf
                warm_starting = False
                print('== Warm Start Finished ==')
                customlog(logfilepath,f"== Warm Start Finished ==\n")
        
            tools['epoch'] = epoch
            if (configs.train_settings.dataloader == "clean" and global_step % total_steps_per_epoch == 0) or configs.train_settings.dataloader != "clean":
               print(f"Fold {valid_batch_number} Epoch {epoch}\n-------------------------------")
               customlog(logfilepath, f"Fold {valid_batch_number} Epoch {epoch} train...\n-------------------------------\n")
            
            start_time = time()

            train_loss = train_loop(tools, configs, warm_starting, train_writer, epoch)
            if configs.train_settings.dataloader != "clean":
               if configs.train_settings.data_aug.enable:
                   tools['train_loader'].dataset.samples = tools['train_loader'].dataset.data_aug_train(tools['train_loader'].dataset.original_samples,configs,tools['train_loader'].dataset.class_weights)
            else: #clean 
               if configs.train_settings.data_aug.enable and global_step % total_steps_per_epoch==0:
                  tools['train_loader'].dataset.samples = tools['train_loader'].dataset.data_aug_train(tools['train_loader'].dataset.original_samples,configs,tools['train_loader'].dataset.class_weights)
            
            train_writer.add_scalar('epoch loss',train_loss,global_step=epoch)
            end_time = time()
        
        
            if epoch % configs.valid_settings.do_every == 0 and epoch != 0:
                customlog(logfilepath, f'Epoch {epoch}: train loss: {train_loss:>5f}\n')
                print(f'Epoch {epoch}: train loss: {train_loss:>5f}\n')
                print(f"Fold {valid_batch_number} Epoch {epoch} validation...\n-------------------------------\n")
                customlog(logfilepath, f"Fold {valid_batch_number} Epoch {epoch} validation...\n-------------------------------\n")
                start_time = time()
                dataloader = tools["valid_loader"]
                
                valid_loss,valid_class_loss,valid_position_loss = test_loop(tools, dataloader,train_writer,valid_writer,configs) #In test loop, never test supcon loss
                
                valid_writer.add_scalar('epoch loss',valid_loss,global_step=epoch)
                valid_writer.add_scalar('epoch class_loss',valid_class_loss,global_step=epoch)
                valid_writer.add_scalar('epoch position_loss',valid_position_loss,global_step=epoch)
                customlog(logfilepath,f'Epoch {epoch}: valid loss:{valid_loss:>5f}\n')
                customlog(logfilepath,f'Epoch {epoch}: valid_class_loss:{valid_class_loss:>5f}\tvalid_position_loss:{valid_position_loss:>5f}\n')
                print(f'Epoch {epoch}: valid loss:{valid_loss:>5f}\n')
                print(f'Epoch {epoch}: valid_class_loss:{valid_class_loss:>5f}\tvalid_position_loss:{valid_position_loss:>5f}\n')
                end_time = time()
            
        
                if warm_starting: #in warm_starting only supcon loss, and train_loss
                  if train_loss < best_valid_loss:
                    customlog(logfilepath, f"Epoch {epoch}: train loss {train_loss} smaller than best loss {best_valid_loss}\n-------------------------------\n")
                    best_valid_loss = train_loss
                    model_path = os.path.join(tools['checkpoint_path'], f'best_model.pth')
                    customlog(logfilepath, f"Epoch {epoch}: A better checkpoint is saved into {model_path} \n-------------------------------\n")
                    save_checkpoint(epoch, model_path, tools)
                else:
                  if valid_loss < best_valid_loss:
                    customlog(logfilepath, f"Epoch {epoch}: valid loss {valid_loss} smaller than best loss {best_valid_loss}\n-------------------------------\n")
                    best_valid_loss = valid_loss
                    model_path = os.path.join(tools['checkpoint_path'], f'best_model.pth')
                    customlog(logfilepath, f"Epoch {epoch}: A better checkpoint is saved into {model_path} \n-------------------------------\n")
                    save_checkpoint(epoch, model_path, tools)
                  if valid_position_loss < best_valid_position_loss:
                      customlog(logfilepath,
                                f"Epoch {epoch}: valid position loss {valid_position_loss} smaller than best position loss {best_valid_position_loss}\n-------------------------------\n")
                      best_valid_position_loss = valid_position_loss
                      model_path1 = os.path.join(tools['checkpoint_path'], f'best_position_model.pth')
                      customlog(logfilepath,
                                f"Epoch {epoch}: A better checkpoint is saved into {model_path1} \n-------------------------------\n")
                      save_checkpoint(epoch, model_path1, tools)

            if epoch % configs.checkpoints_every == 0 and epoch != 0:
                model_path_every = os.path.join(tools['checkpoint_path'], f'checkpoint_{epoch}.pth')
                save_checkpoint(epoch, model_path_every, tools)
    
    if args.predict==1:
       if os.path.exists(configs.resume.resume_path):
          model_path = configs.resume.resume_path
       else:
          model_path = os.path.join(tools['checkpoint_path'], f'best_model.pth')
    
    customlog(logfilepath, f"Loading checkpoint from {model_path}\n")
    model_checkpoint = torch.load(model_path, map_location='cpu')
    tools['net'].load_state_dict(model_checkpoint['model_state_dict'])
    customlog(logfilepath, f"Fold {valid_batch_number} test\n-------------------------------\n")
    start_time = time()
    dataloader = tools["valid_loader"]
    data_dict = get_data_dict(args, dataloader, tools)
    opti_cutoffs_pro, opti_cutoffs_aa = evaluate_protein(data_dict, tools, False)

    dataloader = tools["test_loader"]
    data_dict = get_data_dict(args, dataloader, tools)
    test_protein(data_dict, tools, opti_cutoffs_pro, opti_cutoffs_aa, False)

    dataloader = tools["test_loader"]
    data_dict = get_data_dict(args, dataloader, tools)
    test_protein(data_dict, tools, opti_cutoffs_pro, opti_cutoffs_aa, True)

    customlog(logfilepath, f"\n\n\n\nComparable Test Data\n-------------------------------\n")

    # filter_list = ['Q9LPZ4', 'P15330', 'P35869', 'P51587', 'P70278', 'Q80UP3', 'Q8BMI0', 'Q9HC62', 'Q9UK80',
    #                'Q8LH59', 'P19484', 'P25963', 'P35123', 'Q6NVF4', 'Q8NG08', 'Q9BVS4', 'Q9NRA0', 'Q9NUL5', 'Q9UBP0', 'P78953',
    #                'A8MR65', 'Q8S4Q6', 'Q3U0V2', 'Q96D46', 'Q9NYA1', 'Q9ULX6', 'Q9WTL8', 'Q9WTZ9',
    #                'P35922', 'P46934', 'P81299', 'Q13148', 'Q6ICB0', 'Q7TPV4', 'Q8N884', 'Q99LG4', 'Q9Z207',
    #                'O00571', 'O35973', 'O54943', 'P52306', 'Q13015', 'Q13568', 'Q5TAQ9', 'Q8NAG6', 'Q9BZ23', 'Q9BZS1', 'Q9GZX7',
    #                ]  # 1, 2, 3, 4, 0

    filter_list = [
        'Q9LPZ4', 'P15330', 'P35869', 'P70278', 'Q80UP3',
        'Q8LH59', 'P19484', 'P35123', 'Q6NVF4', 'Q8NG08', 'Q9BVS4', 'Q9NRA0', 'Q9NUL5', 'Q9UBP0', 'P78953',
        'A8MR65', 'Q8S4Q6', 'Q3U0V2', 'Q96D46', 'Q9NYA1', 'Q9ULX6', 'Q9WTL8',
        'P35922', 'P46934', 'P81299', 'Q13148', 'Q6ICB0', 'Q7TPV4', 'Q8N884', 'Q99LG4', 'Q9Z207',
        'O00571', 'P52306', 'Q13015', 'Q13568', 'Q5TAQ9', 'Q8NAG6', 'Q9BZ23', 'Q9BZS1',
    ]

    dataloader = tools["test_loader"]
    data_dict = get_data_dict(args, dataloader, tools)
    filtered_data_dict = {key: value for key, value in data_dict.items() if key in filter_list}
    test_protein(filtered_data_dict, tools, opti_cutoffs_pro, opti_cutoffs_aa, False)

    dataloader = tools["test_loader"]
    data_dict = get_data_dict(args, dataloader, tools)
    filtered_data_dict = {key: value for key, value in data_dict.items() if key in filter_list}
    test_protein(filtered_data_dict, tools, opti_cutoffs_pro, opti_cutoffs_aa, True)

    train_writer.close()
    valid_writer.close()
    end_time = time()
    
    del tools, encoder, dataloaders_dict, optimizer, scheduler
    torch.cuda.empty_cache()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CPM')
    parser.add_argument("--config_path", help="The location of config file", default='./config.yaml')
    parser.add_argument("--predict", type=int, help="predict:1 no training, call evaluate_protein; predict:0 call training loop", default=0)
    parser.add_argument("--result_path", default=None,
                        help="result_path, if setted by command line, overwrite the one in config.yaml, "
                             "by default is None")
    parser.add_argument("--resume_path", default=None,
                        help="if set, overwrite the one in config.yaml, by default is None")

    parser.add_argument("--fold_num", default=1, help="")
    
    args = parser.parse_args()

    config_path = args.config_path

    # print(args.fold_num)
    # exit(0)

    with open(config_path) as file:
        config_dict = yaml.full_load(file)

    for i in range(int(args.fold_num)):
        valid_num = i
        if valid_num == 4:
            test_num = 0
        else:
            test_num = valid_num+1
        main(config_dict, args, valid_num, test_num)
    #     break
    # main(config_dict, args, 0, 1)
    # main(config_dict, args, 1, 2)
    # main(config_dict, args, 2, 3)
    # main(config_dict, args, 3, 4)
    # main(config_dict, args, 4, 0)










# test commit test commit