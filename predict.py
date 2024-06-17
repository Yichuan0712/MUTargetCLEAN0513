import torch
torch.manual_seed(0)
import argparse
import os
import yaml
import numpy as np
from model import *
from utils import *
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import random
from time import time
import json
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 10)
from scipy.ndimage import gaussian_filter
from data_batchsample import *
from data_batchsample import prepare_dataloaders as prepare_dataloader_batchsample
from train import *


def get_prediction(n, data_dict):
    s=len(data_dict.keys())
    result_pro=np.zeros([s,n])
    motif_pred = [{} for i in range(n)]
    result_id = []
    for head in range(n):
        x_list=[]
        for id_protein in data_dict.keys():
            x = data_dict[id_protein]['motif_logits_protein'][head]  #[seq]
            motif_pred[head][id_protein]=x
            x_list.append(data_dict[id_protein]['type_pred'][head])
            if not id_protein in result_id:
                result_id.append(id_protein)
        
        pred=np.array(x_list)
        result_pro[:,head] = pred
    
    return result_pro, motif_pred, result_id

def frag2protein_pred(data_dict, tools):
    overlap=tools['frag_overlap']
    # no_overlap=tools['max_len']-2-overlap
    for id_protein in data_dict.keys():
        id_frag_list = data_dict[id_protein]['id_frag']
        seq_protein=""
        motif_logits_protein=np.array([])
        for i in range(len(id_frag_list)):
            id_frag = id_protein+"@"+str(i)
            ind = id_frag_list.index(id_frag)
            seq_frag = data_dict[id_protein]['seq_frag'][ind]
            motif_logits_frag = data_dict[id_protein]['motif_logits'][ind]
            l=len(seq_frag)
            if i==0:
                seq_protein=seq_frag
                motif_logits_protein=motif_logits_frag[:,:l]
            else:
                seq_protein = seq_protein + seq_frag[overlap:]
                x_overlap = (motif_logits_protein[:,-overlap:] + motif_logits_frag[:,:overlap])/2                
                motif_logits_protein = np.concatenate((motif_logits_protein[:,:-overlap], x_overlap, motif_logits_frag[:,overlap:l]),axis=1)
        data_dict[id_protein]['seq_protein']=seq_protein
        motif_logits_protein[0] = gaussian_filter(motif_logits_protein[0], sigma=2, truncate=1,mode="nearest") # nucleus
        motif_logits_protein[4] = gaussian_filter(motif_logits_protein[4], sigma=2, truncate=1,mode="nearest") # nucleus_export
        data_dict[id_protein]['motif_logits_protein']=motif_logits_protein
    return data_dict

def present(tools, result_pro, motif_pred, result_id):
    classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
             "SIGNAL", "chloroplast", "Thylakoid"]
    output_file = os.path.join(tools['result_path'],"prediction_results.txt")
    logfile=open(output_file, "w")

    cutoffs = list(tools["cutoffs"])
    result_bool = np.zeros_like(result_pro)
    for j in range(result_pro.shape[1]):
        result_bool[:,j] = result_pro[:,j]>cutoffs[j]
    for i in range(len(result_id)):
        id = result_id[i]
        logfile.write(id+"\n")
        pro_pred = result_bool[i]
        pred = np.where(pro_pred==1)[0]
        if pred.size == 0:
            logfile.write("Other\n")
        else:
            for j in pred:
                logfile.write(classname[j]+"\n")
                logfile.write(str(motif_pred[j][id]>cutoffs[j])+"\n")
    logfile.close()

def fix_pred(result_pro, motif_pred, result_id, cutoffs):
    ind_thylakoid = -1
    ind_chlo = -2
    for i in range(result_pro.shape[0]):
        id = result_id[i]
        if result_pro[i,ind_thylakoid]>=cutoffs[ind_thylakoid]:
            result_pro[i,ind_chlo]=1
            cs_thy = np.argmax(motif_pred[ind_thylakoid][id])
            motif_pred[ind_chlo][id][cs_thy:]=0
    return result_pro, motif_pred

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def present2(tools, result_pro, motif_pred, result_id):
    result={}
    classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
             "SIGNAL", "chloroplast", "Thylakoid"]
    cutoffs = list(tools["cutoffs"])

    result_pro, motif_pred = fix_pred(result_pro, motif_pred, result_id, cutoffs)

    for i in range(len(result_id)):
        id = result_id[i]
        seq_len = len(motif_pred[0][id])
        result[id]={}
        for j in range(len(classname)):
            name=classname[j]
            if result_pro[i,j]<cutoffs[j]:
                result[id][name]=""
            else:
                if name in ["Mitochondrion","SIGNAL", "chloroplast", "Thylakoid"]:
                    result[id][name]="0-"+ str(np.argmax(motif_pred[j][id]))
                elif name == "ER":
                    result[id][name]=str(np.argmax(motif_pred[j][id])) + "-"+ str(seq_len-1)
                elif name == "Peroxisome":
                    cs = np.argmax(motif_pred[j][id])
                    if seq_len-cs > cs:
                        result[id][name]="0-"+ str(cs)
                    else:
                        result[id][name]=str(cs) + "-"+ str(seq_len-1)
                elif name in ["Nucleus", "Nucleus_export"]:
                    sites= np.where(motif_pred[j][id]>cutoffs[j])[0]
                    if len(sites)==0:
                        site = np.argmax(motif_pred[j][id])
                        sites = [site]
                        if site-2>=0:
                            sites = [site-2, site-1, site]
                        if site+2<=seq_len-1:
                            sites = sites.extend([site+1, site+2])
                    result[id][name]=str(np.array(sites))
    json_object = json.dumps(result, indent=2, cls=NpEncoder)
    output_file = os.path.join(tools['result_path'],"prediction_results.json")
    with open(output_file, "w") as outfile:
        outfile.write(json_object)

def present3(tools, result_pro, motif_pred, result_id, data_dict):
    id2AAscores={}
    id2label={}
    classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
             "SIGNAL", "chloroplast", "Thylakoid"]
    AA_cutoffs = list(tools["AA_cutoffs"])
    prot_cutoffs = list(tools["prot_cutoffs"])

    result_pro, motif_pred = fix_pred(result_pro, motif_pred, result_id, prot_cutoffs)

    for i in range(len(result_id)):
        id = result_id[i]
        seq = data_dict[id]['seq_protein']
        seq_len = len(seq)
        motif_score = np.zeros(seq_len)
        id2AAscores[id]=motif_score
        label=""
        for j in range(len(classname)):
            name=classname[j]
            if result_pro[i,j]<prot_cutoffs[j]:
                continue
            else:
                if label=="":
                    label=name
                else:
                    label+=", "+name
                score = j+1
                if name in ["Mitochondrion","SIGNAL", "chloroplast", "Thylakoid"]:
                    # result[id][name]="0-"+ str(np.argmax(motif_pred[j][id]))
                    cs = np.argmax(motif_pred[j][id])
                    for k in range(cs+1):
                        if motif_score[k]==0:
                            motif_score[k]=score
                elif name == "ER":
                    # result[id][name]=str(np.argmax(motif_pred[j][id])) + "-"+ str(seq_len-1)
                    cs = np.argmax(motif_pred[j][id])
                    motif_score[cs:seq_len]=score
                elif name == "Peroxisome":
                    cs = np.argmax(motif_pred[j][id])
                    if seq_len-cs > cs:
                        # result[id][name]="0-"+ str(cs)
                        motif_score[0:cs+1]=score
                    else:
                        # result[id][name]=str(cs) + "-"+ str(seq_len-1)
                        motif_score[cs:seq_len]=score
                elif name in ["Nucleus", "Nucleus_export"]:
                    sites= np.where(motif_pred[j][id]>AA_cutoffs[j])[0]
                    if len(sites)==0:
                        site = np.argmax(motif_pred[j][id])
                        sites = [site]
                        if site-2>=0:
                            sites = [site-2, site-1, site]
                        if site+2<=seq_len-1:
                            sites.extend([site+1, site+2])
                    motif_score[sites]=score
                    # result[id][name]=str(np.array(sites))
        id2AAscores[id]=motif_score
        if label=="":
            label="Others"
        id2label[id]=label
    # json_object = json.dumps(result, indent=2, cls=NpEncoder)
    output_file = os.path.join(tools['result_path'],"prediction_results.txt")
    output_file2 = os.path.join(tools['result_path'],"prediction_scores.txt")
    with open(output_file, "w") as outfile:
        # outfile.write(json_object)
        # for k in result.keys():
        #     outfile.write(">"+str(k)+"\n")
        #     outfile.write(str(result[k]))
        #     outfile.write("\n")
        for i in range(len(result_id)):
            id = result_id[i]
            outfile.write(">"+str(id)+"\n")
            outfile.write("Predictions: " + id2label[id]+ "\n")
            outfile.write("Probabilities: ")
            for j in range(result_pro.shape[1]-1):
                outfile.write(classname[j]+"("+str(j+1)+"): "+str(result_pro[i, j])+", ")
            outfile.write(classname[j+1]+"("+str(j+1+1)+"): "+str(result_pro[i, j+1]))
            outfile.write("\n")
            # outfile.write(str(result_pro[i])+"\n")
            outfile.write(str(id2AAscores[id]))
            outfile.write("\n")

    with open(output_file2, "w") as outfile:
        for i in range(len(result_id)):
            id = result_id[i]
            outfile.write(">"+str(id)+"\n")
            if "Nucleus_export" in id2label[id]:
                outfile.write("Nucleus_export\n")
                outfile.write(str(motif_pred[4][id]))
                outfile.write("\n")
            elif "Nucleus" in id2label[id]:
                outfile.write("Nucleus\n")
                outfile.write(str(motif_pred[0][id]))
                outfile.write("\n")

            




def make_buffer_pred(id_frag_list_tuple, seq_frag_list_tuple):
    id_frags_list = []
    seq_frag_list = []
    for i in range(len(id_frag_list_tuple)):
        id_frags_list.extend(id_frag_list_tuple[i])
        seq_frag_list.extend(seq_frag_list_tuple[i])
    seq_frag_tuple = tuple(seq_frag_list)
    return id_frags_list, seq_frag_tuple

def predict(dataloader, tools):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.eval().cuda()
    tools['net'].eval().to(tools["pred_device"])
    n=tools['num_classes']

    # cutoff = tools['cutoff']
    data_dict={}
    with torch.no_grad():
        for batch, (id_tuple, id_frag_list_tuple, seq_frag_list_tuple) in enumerate(dataloader):
            start_time = time()
            id_frags_list, seq_frag_tuple = make_buffer_pred(id_frag_list_tuple, seq_frag_list_tuple)
            encoded_seq=tokenize(tools, seq_frag_tuple)
            if type(encoded_seq)==dict:
                for k in encoded_seq.keys():
                    encoded_seq[k]=encoded_seq[k].to(tools['pred_device'])
            else:
                encoded_seq=encoded_seq.to(tools['pred_device'])

            classification_head, motif_logits = tools['net'](encoded_seq, id_tuple, id_frags_list, seq_frag_tuple)
            m=torch.nn.Sigmoid()
            motif_logits = m(motif_logits)
            classification_head = m(classification_head)

            x_frag = np.array(motif_logits.cpu())   #[batch, head, seq]
            x_pro = np.array(classification_head.cpu()) #[sample, n]
            for i in range(len(id_frags_list)):
                id_protein=id_frags_list[i].split('@')[0]
                j= id_tuple.index(id_protein)
                if id_protein in data_dict.keys():
                    data_dict[id_protein]['id_frag'].append(id_frags_list[i])
                    data_dict[id_protein]['seq_frag'].append(seq_frag_tuple[i])
                    data_dict[id_protein]['motif_logits'].append(x_frag[i])    #[[head, seq], ...]
                else:
                    data_dict[id_protein]={}
                    data_dict[id_protein]['id_frag']=[id_frags_list[i]]
                    data_dict[id_protein]['seq_frag']=[seq_frag_tuple[i]]
                    data_dict[id_protein]['motif_logits']=[x_frag[i]]
                    data_dict[id_protein]['type_pred']=x_pro[j]
            end_time = time()
            print("One batch time: "+ str(end_time-start_time))

        start_time = time()
        data_dict = frag2protein_pred(data_dict, tools)
        end_time = time()
        print("frag ensemble time: "+ str(end_time-start_time))

        start_time = time()
        result_pro, motif_pred, result_id = get_prediction(n, data_dict)  # result_pro = [sample_size, class_num], sample order same as result_id  
                                                                                         # motif_pred = class_num dictionaries with protein id as keys
                                                                                         # result_id = [sample_size], protein ids
        end_time = time()
        print("get prediction time: "+ str(end_time-start_time))
        # present(tools, result_pro, motif_pred, result_id)

        start_time = time()
        present3(tools, result_pro, motif_pred, result_id, data_dict)
        end_time = time()
        print("present time: "+ str(end_time-start_time))

def get_scores_pred(tools, n, data_dict, constrain):
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
    prot_cutoffs = list(tools["prot_cutoffs"])
    AA_cutoffs = list(tools["AA_cutoffs"])

    # TPR_pro=np.zeros(n)
    # FPR_pro=np.zeros(n)
    # FNR_pro=np.zeros(n)
    IoU_pro = np.zeros(n)
    # Negtive_detect_pro=0
    # Negtive_pro=0
    result_pro=np.zeros([n,6])
    for head in range(n):
        x_list=[]
        y_list=[]
        for id_protein in data_dict.keys():
            x_pro = data_dict[id_protein]['type_pred'][head]  #[1]
            y_pro = data_dict[id_protein]['type_target'][head]  #[1]   
            x_list.append(x_pro)  
            y_list.append(y_pro)
            if constrain:
                condition = x_pro >= prot_cutoffs[head]
            else:
                condition = True
            if y_pro == 1 and condition:
                x_frag = data_dict[id_protein]['motif_logits_protein'][head]  #[seq]
                y_frag = data_dict[id_protein]['motif_target_protein'][head]
                # Negtive_pro += np.sum(np.max(y)==0)
                # Negtive_detect_pro += np.sum((np.max(y)==0) * (np.max(x>=cutoff)==1))
                TPR_pro = np.sum((x_frag>=AA_cutoffs[head]) * (y_frag==1))/np.sum(y_frag==1)
                FPR_pro = np.sum((x_frag>=AA_cutoffs[head]) * (y_frag==0))/np.sum(y_frag==0)
                FNR_pro = np.sum((x_frag<AA_cutoffs[head]) * (y_frag==1))/np.sum(y_frag==1)
                # x_list.append(np.max(x))
                # y_list.append(np.max(y))
                IoU_pro[head] += TPR_pro / (TPR_pro + FPR_pro + FNR_pro)
    
                cs_num[head] += np.sum(y_frag==1)>0
                if np.sum(y_frag==1)>0:
                    cs_correct[head] += (np.argmax(x_frag) == np.argmax(y_frag))

        IoU_pro[head] = IoU_pro[head] / sum(y_list)
        cs_acc[head] = cs_correct[head] / cs_num[head]

        pred=np.array(x_list)
        target=np.array(y_list)
        try:
            result_pro[head,0] = roc_auc_score(target, pred)
            result_pro[head,1] = average_precision_score(target, pred)
            result_pro[head,2] = matthews_corrcoef(target, pred>=prot_cutoffs[head])
            result_pro[head,3] = recall_score(target, pred>=prot_cutoffs[head])
            result_pro[head,4] = precision_score(target, pred>=prot_cutoffs[head])
            result_pro[head,5] = f1_score(target, pred>=prot_cutoffs[head])
        except Exception as e:
            print("error is: ", e)
        
    # for head in range(n):
    #     # IoU[head] = TP_frag[head] / (TP_frag[head] + FP_frag[head] + FN_frag[head])
    #     IoU_pro[head] = TPR_pro[head] / (TPR_pro[head] + FPR_pro[head] + FNR_pro[head])
    #     cs_acc[head] = cs_correct[head] / cs_num[head]
    # FDR_frag = Negtive_detect_num / Negtive_num
    # FDR_pro = Negtive_detect_pro / Negtive_pro
    
    scores={"IoU_pro":IoU_pro, #[n]
            "result_pro":result_pro, #[n, 6]
            "cs_acc": cs_acc} #[n]
    return scores

def get_evaluation(tools, data_dict, constrain):
    n=tools['num_classes']
    # IoU_pro_difcut=np.zeros([n, 9])  #just for nuc and nuc_export
    IoU_pro_difcut=np.zeros([n])  #just for nuc and nuc_export
    # result_pro_difcut=np.zeros([n,6,9])
    result_pro_difcut=np.zeros([n,6])
    # cs_acc_difcut=np.zeros([n, 9]) 
    cs_acc_difcut=np.zeros([n]) 
    classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
         "SIGNAL", "chloroplast", "Thylakoid"]
    criteria=["roc_auc_score", "average_precision_score", "matthews_corrcoef",
          "recall_score", "precision_score", "f1_score"]
    # cutoffs=[x / 10 for x in range(1, 10)]
    # cut_dim=0
    # for cutoff in cutoffs:
    scores=get_scores_pred(tools, n, data_dict, constrain)
    IoU_pro_difcut=scores['IoU_pro']
    result_pro_difcut=scores['result_pro']
    cs_acc_difcut=scores['cs_acc'] 
        # cut_dim+=1
    evaluation_file = os.path.join(tools['result_path'],"evaluation.txt")
    customlog(evaluation_file, f"===========================================\n")
    customlog(evaluation_file, f" Jaccard Index (protein): \n")
    # IoU_pro_difcut=pd.DataFrame(IoU_pro_difcut, index=classname)
    IoU_pro_difcut=pd.DataFrame([IoU_pro_difcut[i] for i in [0,4]], index=[classname[i] for i in [0,4]])
    customlog(evaluation_file, IoU_pro_difcut.__repr__())
    customlog(evaluation_file, f"===========================================\n")
    customlog(evaluation_file, f" cs acc: \n")
    # cs_acc_difcut=pd.DataFrame(cs_acc_difcut, index=classname)
    cs_acc_difcut=pd.DataFrame([cs_acc_difcut[i] for i in [1,2,3,5,6,7]], index=[classname[i] for i in [1,2,3,5,6,7]])
    customlog(evaluation_file, cs_acc_difcut.__repr__())
    customlog(evaluation_file, f"===========================================\n")
    # for i in range(len(classname)):
    #     customlog(evaluation_file, f" Class prediction performance ({classname[i]}): \n")
    #     tem = pd.DataFrame(result_pro_difcut[i], index=criteria)
    #     customlog(evaluation_file, tem.__repr__())
    tem = pd.DataFrame(result_pro_difcut, columns=criteria, index=classname)
    customlog(evaluation_file, tem.__repr__())

def get_individual_scores(tools, data_dict, constrain):
    n=tools['num_classes']
    # cutoffs = list(tools["cutoffs"])
    AA_cutoffs = list(tools["AA_cutoffs"])
    prot_cutoffs = list(tools["prot_cutoffs"])
    classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
             "SIGNAL", "chloroplast", "Thylakoid"]
    evaluation_file = os.path.join(tools['result_path'],"evaluation_individual.txt")
    customlog(evaluation_file, f"ID\tlabel\tpredict\tIoU\tcs_correct\tmotif")
    for head in range(n):
        name=classname[head]
        x_list=[]
        y_list=[]
        AA_cutoff = AA_cutoffs[head]
        prot_cutoff = prot_cutoffs[head]
        for id_protein in data_dict.keys():
            seq = data_dict[id_protein]['seq_protein']
            seq_len = len(seq)
            x_pro = data_dict[id_protein]['type_pred'][head]  #[1]
            y_pro = data_dict[id_protein]['type_target'][head]  #[1]   
            x_list.append(x_pro)  
            y_list.append(y_pro)
            if constrain:
                condition = x_pro >= prot_cutoff
            else:
                condition = True
            if y_pro == 1 and condition:
                x_frag = data_dict[id_protein]['motif_logits_protein'][head]  #[seq]
                y_frag = data_dict[id_protein]['motif_target_protein'][head]
                # Negtive_pro += np.sum(np.max(y)==0)
                # Negtive_detect_pro += np.sum((np.max(y)==0) * (np.max(x>=cutoff)==1))
                TPR_pro = np.sum((x_frag>=AA_cutoff) * (y_frag==1))/np.sum(y_frag==1)
                FPR_pro = np.sum((x_frag>=AA_cutoff) * (y_frag==0))/np.sum(y_frag==0)
                FNR_pro = np.sum((x_frag<AA_cutoff) * (y_frag==1))/np.sum(y_frag==1)
                # x_list.append(np.max(x))
                # y_list.append(np.max(y))
                IoU_pro = TPR_pro / (TPR_pro + FPR_pro + FNR_pro)

                cs_correct = (np.argmax(x_frag) == np.argmax(y_frag))

                motif=""
                if name in ["Mitochondrion","SIGNAL","chloroplast","Thylakoid"]:
                    cs = np.argmax(x_frag)
                    motif=seq[0:cs+1]
                elif name =="ER":
                    cs = np.argmax(x_frag)
                    motif=seq[cs:]
                elif name =="Peroxisome":
                    cs = np.argmax(x_frag)
                    if seq_len-cs>cs:
                        motif = seq[0:cs+1]
                    else:
                        motif = seq[cs:]
                elif name in ["Nucleus","Nucleus_export"]:
                    sites = np.where(x_frag>AA_cutoff)[0]
                    if len(sites)==0:
                        site = np.argmax(x_frag)
                        sites = [site]
                        if site-2>=0:
                            sites = [site-2, site-1, site]
                        if site+2<=seq_len-1:
                            sites.extend([site+1, site+2])
                    sites = np.array(sites)
                    motif = ''.join([seq[i] for i in sites])
                customlog(evaluation_file, f"{id_protein}\t{name}\t{x_pro}\t{IoU_pro}\t{cs_correct}\t{motif}\n")
                


def predict_withlabel(dataloader, tools, constrain):
    tools['net'].eval().to(tools["pred_device"])
    n=tools['num_classes']

    # cutoff = tools['cutoff']
    data_dict={}
    with torch.no_grad():
        for batch, (id_tuple, id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple, sample_weight_tuple, pos_neg) in enumerate(dataloader):
            start_time = time()
            id_frags_list, seq_frag_tuple, target_frag_pt, type_protein_pt = make_buffer(id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple)
            encoded_seq=tokenize(tools, seq_frag_tuple)
            if type(encoded_seq)==dict:
                for k in encoded_seq.keys():
                    encoded_seq[k]=encoded_seq[k].to(tools['pred_device'])
            else:
                encoded_seq=encoded_seq.to(tools['pred_device'])

            # classification_head, motif_logits = tools['net'](encoded_seq, id_tuple, id_frags_list, seq_frag_tuple)
            classification_head, motif_logits, projection_head = tools['net'](
                       encoded_seq,
                       id_tuple,id_frags_list,seq_frag_tuple, 
                       None, False) #for test_loop always used None and False!

            m=torch.nn.Sigmoid()
            motif_logits = m(motif_logits)
            classification_head = m(classification_head)

            x_frag = np.array(motif_logits.cpu())   #[batch, head, seq]
            x_pro = np.array(classification_head.cpu()) #[sample, n]
            y_frag = np.array(target_frag_pt.cpu())    #[batch, head, seq]
            y_pro = np.array(type_protein_pt.cpu()) #[sample, n]
            for i in range(len(id_frags_list)):
                id_protein=id_frags_list[i].split('@')[0]
                j= id_tuple.index(id_protein)
                if id_protein in data_dict.keys():
                    data_dict[id_protein]['id_frag'].append(id_frags_list[i])
                    data_dict[id_protein]['seq_frag'].append(seq_frag_tuple[i])
                    data_dict[id_protein]['target_frag'].append(y_frag[i])     #[[head, seq], ...]
                    data_dict[id_protein]['motif_logits'].append(x_frag[i])    #[[head, seq], ...]
                else:
                    data_dict[id_protein]={}
                    data_dict[id_protein]['id_frag']=[id_frags_list[i]]
                    data_dict[id_protein]['seq_frag']=[seq_frag_tuple[i]]
                    data_dict[id_protein]['target_frag']=[y_frag[i]]
                    data_dict[id_protein]['motif_logits']=[x_frag[i]]
                    data_dict[id_protein]['type_pred']=x_pro[j]
                    data_dict[id_protein]['type_target']=y_pro[j]
            end_time = time()
            print("One batch time: "+ str(end_time-start_time))
        print("!@# data_dict size "+str(len(data_dict)))

        start_time = time()
        data_dict = frag2protein(data_dict, tools)
        end_time = time()
        print("frag ensemble time: "+ str(end_time-start_time))

        start_time = time()
        result_pro, motif_pred, result_id = get_prediction(n, data_dict)  # result_pro = [sample_size, class_num], sample order same as result_id  
                                                                                         # motif_pred = class_num dictionaries with protein id as keys
                                                                                         # result_id = [sample_size], protein ids
        print("!@# result id size "+str(len(result_id)))

        end_time = time()
        print("get prediction time: "+ str(end_time-start_time))
        # present(tools, result_pro, motif_pred, result_id)

        get_evaluation(tools, data_dict, constrain)
        print("!@# data_dict size "+str(len(data_dict)))
        get_individual_scores(tools, data_dict, constrain)

        start_time = time()
        present3(tools, result_pro, motif_pred, result_id, data_dict)
        end_time = time()
        print("present time: "+ str(end_time-start_time))
        



        
class LocalizationDataset_pred(Dataset):
    def __init__(self, samples, configs):
        # self.label_to_index = {"Other": 0, "SP": 1, "MT": 2, "CH": 3, "TH": 4}
        # self.index_to_label = {0: "Other", 1: "SP", 2: "MT", 3: "CH", 4: "TH"}
        # self.transform = transform
        # self.target_transform = target_transform
        # self.cs_transform = cs_transform
        self.samples = samples
        self.n = configs.encoder.num_classes
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        id, id_frag_list, seq_frag_list = self.samples[idx]
        return id, id_frag_list, seq_frag_list

def custom_collate_pred(batch):
    id, id_frags, fragments = zip(*batch)
    return id, id_frags, fragments

def split_protein_sequence_pred(prot_id, sequence, configs):
    fragment_length = configs.encoder.max_len - 2
    overlap = configs.encoder.frag_overlap
    fragments = []
    id_frags = []
    sequence_length = len(sequence)
    start = 0
    ind=0

    while start < sequence_length:
        end = start + fragment_length
        if end > sequence_length:
            end = sequence_length
        fragment = sequence[start:end]
        fragments.append(fragment)
        id_frags.append(prot_id+"@"+str(ind))
        ind+=1
        if start + fragment_length > sequence_length:
            break
        start += fragment_length - overlap

    return id_frags, fragments

def prepare_samples_pred(csv_file, configs):
    # label2idx = {"Nucleus":0, "ER":1, "Peroxisome":2, "Mitochondrion":3, "Nucleus_export":4,
    #              "dual":5, "SIGNAL":6, "chloroplast":7, "Thylakoid":8}
    label2idx = {"Nucleus":0, "ER":1, "Peroxisome":2, "Mitochondrion":3, "Nucleus_export":4,
                 "SIGNAL":5, "chloroplast":6, "Thylakoid":7}
    samples = []
    n = configs.encoder.num_classes
    df = pd.read_csv(csv_file)
    row,col=df.shape
    for i in range(row):
        prot_id = df.loc[i,"Entry"]
        seq = df.loc[i,"Sequence"]
  
        id_frag_list, seq_frag_list = split_protein_sequence_pred(prot_id, seq, configs)
        samples.append((prot_id, id_frag_list, seq_frag_list))
        # for j in range(len(seq_frag_list )):
        #     id=prot_id+"@"+str(j)
        #     samples.append((id, fragments[j]))
        
    return samples

def prepare_dataloaders_pred(configs, input_file, with_label):
    # id_to_seq = prot_id_to_seq(seq_file)
    random.seed(configs.fix_seed)
    if with_label:
        samples = prepare_samples(input_file,configs)
        random.shuffle(samples)
        dataset = LocalizationDataset(samples, configs=configs,mode = "test")
        pred_dataloader = DataLoader(dataset, batch_size=configs.predict_settings.batch_size, shuffle=True, collate_fn=custom_collate)
    else:
        samples = prepare_samples_pred(input_file,configs)
        random.shuffle(samples)
        dataset = LocalizationDataset_pred(samples, configs=configs)
        pred_dataloader = DataLoader(dataset, batch_size=configs.predict_settings.batch_size, shuffle=False, collate_fn=custom_collate_pred)
    
    # Shuffle the list
    # print(train_dataset)
    return pred_dataloader


def prepare_pred_dir(configs, output_dir, model_dir):
    curdir_path=os.getcwd()
    checkpoint_file = os.path.abspath(model_dir)
    result_path = os.path.abspath(output_dir)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    return result_path, checkpoint_file, curdir_path


def main(config_dict, input_file, output_dir, model_dir, with_label, constrain):
    configs = load_configs(config_dict, None)
    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()

    start_time = time()
    dataloader = prepare_dataloaders_pred(configs, input_file, with_label)
    end_time = time()
    print("dataloader time: "+ str(end_time-start_time))
    print("!@# dataloader size  "+str(len(dataloader.dataset)))

    result_path, checkpoint_file, curdir_path = prepare_pred_dir(configs, output_dir, model_dir)

    start_time = time()
    tokenizer=prepare_tokenizer(configs, curdir_path)
    encoder=prepare_models(configs, '', curdir_path)
    model_checkpoint = torch.load(checkpoint_file, map_location='cpu')
    encoder.load_state_dict(model_checkpoint['model_state_dict'])
    end_time = time()
    print("model loading time: "+ str(end_time-start_time))

    tools = {
        'frag_overlap': configs.encoder.frag_overlap,
        'AA_cutoffs': configs.predict_settings.AA_cutoffs,
        'prot_cutoffs': configs.predict_settings.prot_cutoffs,
        'composition': configs.encoder.composition, 
        'max_len': configs.encoder.max_len,
        'tokenizer': tokenizer,
        'prm4prmpro': configs.encoder.prm4prmpro,
        'net': encoder,
        'pred_device': configs.predict_settings.device,
        'pred_batch_size': configs.predict_settings.batch_size,
        'result_path': result_path,
        'num_classes': configs.encoder.num_classes
    }
    
    start_time = time()
    if with_label:
        predict_withlabel(dataloader, tools, constrain)
    else:
        predict(dataloader, tools)
    end_time = time()
    print("predicting time: "+ str(end_time-start_time))

    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CPM')
    parser.add_argument("--config_path", help="The location of config file", default='./config.yaml')
    parser.add_argument("--input_file", help="The location of input fasta file")
    parser.add_argument("--output_dir", help="The dir location of output")
    parser.add_argument("--model_dir", help="The dir location of trained model")
    parser.add_argument('--with_label', action='store_true', help='Set the flag to true')
    parser.add_argument('--constrain', action='store_true', help='If use classification cutoff as constrain')
    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path) as file:
        config_dict = yaml.full_load(file)
    
    input_file = args.input_file
    output_dir = args.output_dir
    model_dir = args.model_dir
    with_label = args.with_label
    constrain = args.constrain
    

    main(config_dict, input_file, output_dir, model_dir, with_label, constrain)

#use case

#python predict.py --config_path ./results_log/D0614_2T2loss/P02_5/config0614_E15_T02_T5.yaml --input_file ./test_data_EC269_fold1.csv --output_dir ./results_log/D0614_2T2loss/P02_5/results_fold0 --model_dir ./results_log/D0614_2T2loss/P02_5/best_model_02_5.pth --with_label --constrain#python predict.py --config_path ./results_log/D0614_2T2loss/P02_5/config0614_E15_T02_T5.yaml --input_file ./test_data_1_ECO269_nodupliTargetp.csv --output_dir ./results_log/D0614_2T2loss/P02_5/results_fold0 --model_dir ./results_log/D0614_2T2loss/P02_5/best_model_02_5.pth --with_label --constrain
#python predict.py --config_path ./results_log/D0614_2T2loss/P02_5/config0614_E15_T02_T5.yaml --input_file ./test_data_1_ECO269_nodupliTargetp.csv --output_dir ./results_log/D0614_2T2loss/P02_5/results_fold0 --model_dir ./results_log/D0614_2T2loss/P02_5/best_model_02_5.pth --with_label --constrain#python predict.py --config_path ./results_log/D0614_2T2loss/P02_5/config0614_E15_T02_T5.yaml --input_file ./test_data_1_ECO269_nodupliTargetp.csv --output_dir ./results_log/D0614_2T2loss/P02_5/results_fold0 --model_dir ./results_log/D0614_2T2loss/P02_5/best_model_02_5.pth --with_label --constrain