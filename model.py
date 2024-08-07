from peft import get_peft_model
from peft import LoraConfig, PromptTuningConfig, PromptTuningInit
import torch
from torch import nn

torch.manual_seed(0)
import os
from transformers import AutoTokenizer
from PromptProtein.utils import PromptConverter
from PromptProtein.models import openprotein_promptprotein
from transformers import EsmModel
from utils import customlog, prepare_saving_dir


# from train import make_buffer


class LayerNormNet(nn.Module):
    """
    From https://github.com/tttianhao/CLEAN
    """

    def __init__(self, configs, hidden_dim=512, out_dim=256):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = configs.supcon.drop_out
        self.device = configs.train_settings.device
        self.dtype = torch.float32
        feature_dim = {"facebook/esm2_t6_8M_UR50D": 320, "facebook/esm2_t33_650M_UR50D": 1280,
                       "facebook/esm2_t30_150M_UR50D": 640, "facebook/esm2_t12_35M_UR50D": 480}
        self.fc1 = nn.Linear(feature_dim[configs.encoder.model_name], hidden_dim, dtype=self.dtype, device=self.device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=self.dtype, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=self.dtype, device=self.device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=self.dtype, device=self.device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=self.dtype, device=self.device)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        # x = self.dropout(self.ln2(self.fc2(x)))
        # x = torch.relu(x)
        x = self.fc3(x)
        return x


def initialize_PromptProtein(pretrain_loc, trainable_layers):
    from PromptProtein.models import openprotein_promptprotein
    from PromptProtein.utils import PromptConverter
    model, dictionary = openprotein_promptprotein(pretrain_loc)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        for lname in trainable_layers:
            if lname in name:
                param.requires_grad = True
    return model


class CustomPromptModel(nn.Module):
    def __init__(self, configs, pretrain_loc, trainable_layers):
        super(CustomPromptModel, self).__init__()
        self.pretrain = initialize_PromptProtein(pretrain_loc, trainable_layers)
        # self.learnable_prompt_emb = nn.Parameter(initialize_learnable_prompts(prompt_tok_vec, self.pretrain, self.converter)).to("cuda")
        # self.learnable_prompt_emb = nn.Parameter(initialize_learnable_prompts(8, self.pretrain, self.converter)).to("cuda")
        # self.learnable_prompt_emb = nn.Parameter(initialize_learnable_prompts(8, self.pretrain, self.converter, self.device)).to(self.device)
        # self.decoder_class = Decoder_linear(input_dim=1280, output_dim=5)
        self.pooling_layer = nn.AdaptiveAvgPool1d(output_size=1)
        self.head = nn.Linear(1280, configs.encoder.num_classes)
        self.cs_head = nn.Linear(1280, 1)
        # if decoder_cs=="linear":
        #     self.decoder_cs = Decoder_linear(input_dim=1280, output_dim=1)

    def forward(self, encoded_seq):
        result = self.pretrain(encoded_seq, with_prompt_num=1)
        # logits size => (B, T+2, E)
        logits = result['logits']

        transposed_feature = logits.transpose(1, 2)
        pooled_features = self.pooling_layer(transposed_feature).squeeze(2)
        type_probab = self.head(pooled_features)
        cs_head = self.cs_head(logits).squeeze(dim=-1)
        cs_pred = cs_head[:, 1:]
        # print("peptide_probab size ="+str(type_probab.size()))
        return type_probab, cs_pred


def prepare_tokenizer(configs, curdir_path):
    if configs.encoder.composition == "esm_v2":
        tokenizer = AutoTokenizer.from_pretrained(configs.encoder.model_name)
    elif configs.encoder.composition == "promprot":
        model, dictionary = openprotein_promptprotein(os.path.join(curdir_path, "PromptProtein", "PromptProtein.pt"))
        tokenizer = PromptConverter(dictionary)
    elif configs.encoder.composition == "both":
        tokenizer_esm = AutoTokenizer.from_pretrained(configs.encoder.model_name)
        model, dictionary = openprotein_promptprotein(os.path.join(curdir_path, "PromptProtein", "PromptProtein.pt"))
        tokenizer_promprot = PromptConverter(dictionary)
        tokenizer = {"tokenizer_esm": tokenizer_esm, "tokenizer_promprot": tokenizer_promprot}
    else:
        tokenizer = None  # for adapter
    return tokenizer


def tokenize(tools, seq):
    if tools['composition'] == "esm_v2":
        max_length = tools['max_len']
        encoded_sequence = tools["tokenizer"](seq, max_length=max_length, padding='max_length',
                                              truncation=True,
                                              return_tensors="pt"
                                              )
        # encoded_sequence['input_ids'] = torch.squeeze(encoded_sequence['input_ids'])
        # encoded_sequence['attention_mask'] = torch.squeeze(encoded_sequence['attention_mask'])
    elif tools['composition']=="official_esm_v2":
        # data = [("", one_seq) for one_seq in seq]

        data = []
        for one_seq in seq:
            if len(one_seq) < tools['max_len']:
                one_seq = one_seq + "<pad>" * (tools['max_len'] - len(one_seq) - 2)
            data.append(("", one_seq))

        _, _, encoded_sequence = tools["tokenizer"](data)
    elif tools['composition'] == "promprot":
        if tools['prm4prmpro'] == 'seq':
            prompts = ['<seq>']
            encoded_sequence = tools["tokenizer"](seq, prompt_toks=prompts)
        elif tools['prm4prmpro'] == 'ppi':
            prompts = ['<ppi>']
            encoded_sequence = tools["tokenizer"](seq, prompt_toks=prompts)
    elif tools['composition'] == "both":
        max_length = tools['max_len']
        encoded_sequence_esm2 = tools["tokenizer"]["tokenizer_esm"](seq, max_length=max_length, padding='max_length',
                                                                    truncation=True,
                                                                    return_tensors="pt"
                                                                    )
        if tools['prm4prmpro'] == 'seq':
            prompts = ['<seq>']
            encoded_sequence_promprot = tools["tokenizer"]["tokenizer_promprot"](seq, prompt_toks=prompts)
        elif tools['prm4prmpro'] == 'ppi':
            prompts = ['<ppi>']
            encoded_sequence_promprot = tools["tokenizer"]["tokenizer_promprot"](seq, prompt_toks=prompts)
        encoded_sequence = {"encoded_sequence_esm2": encoded_sequence_esm2,
                            "encoded_sequence_promprot": encoded_sequence_promprot}
    return encoded_sequence


def print_trainable_parameters(model, logfilepath):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    customlog(logfilepath,
              f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}\n")


def prepare_esm_model(model_name, configs):
    model = EsmModel.from_pretrained(model_name)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    if configs.PEFT == "lora":
        config = LoraConfig(target_modules=["query", "key"])
        model = get_peft_model(model, config)
    # elif configs.PEFT == "PromT":
    #     config = PromptTuningConfig(task_type="SEQ_CLS", prompt_tuning_init=PromptTuningInit.TEXT, num_virtual_tokens=8,
    #                             prompt_tuning_init_text="Classify what the peptide type of a protein sequence", tokenizer_name_or_path=configs.encoder.model_name)
    #     model = get_peft_model(model, config)
    #     for param in model.encoder.layer[-1].parameters():
    #         param.requires_grad = True
    #     for param in model.pooler.parameters():
    #         param.requires_grad = False
    elif configs.PEFT == "frozen":
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
    elif configs.PEFT == "PFT":
        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.encoder.layer[configs.train_settings.fine_tune_lr:].parameters():
            param.requires_grad = True
        for param in model.pooler.parameters():
            param.requires_grad = False
    elif configs.PEFT == "lora_PFT":
        config = LoraConfig(target_modules=["query", "key"])
        model = get_peft_model(model, config)
        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.encoder.layer[configs.train_settings.fine_tune_lr:].parameters():
            param.requires_grad = True
        for param in model.pooler.parameters():
            param.requires_grad = False
    return model


def initialize_weights(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)  # Initialize biases to zero


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, droprate=0.3):
        super(SimpleCNN, self).__init__()
        self.kernel_size = kernel_size
        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_layer2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        # self.linear = nn.Linear(out_channels, 1)
        """
        #Initialize weights from a uniform distribution within the specified range
        weight_range = torch.sqrt(torch.tensor(1.0 / in_channels))  # Range for weight initialization
        #self.conv_layer.weight.data.uniform_(-weight_range, weight_range)
        nn.init.uniform_(self.conv_layer.weight, -weight_range.item(), weight_range.item())
        # Initialize biases to zeros
        nn.init.zeros_(self.conv_layer.bias)
        """
        # self.conv_layer.apply(initialize_weights)
        # Use nn.Linear's default initialization for conv1d weights
        # self.conv_layer.reset_parameters()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.dropout(x)
        x = self.conv_layer2(x)
        # x = self.activation_spread(x)
        # x = self.dropout(x)
        # x = self.relu(x)  #relu cannot be used with sigmoid!!! smallest will be 0.5
        x, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling across output channels
        x = x.squeeze(1)
        # x  = self.linear(x.permute(0,2,1)).squeeze(-1)
        return x

    def activation_spread(self, x):
        if self.kernel_size % 2 == 0:
            pad_left = (self.kernel_size // 2) - 1
            pad_right = self.kernel_size // 2
        else:
            pad_left = pad_right = (self.kernel_size - 1) // 2

        # x is the input activation map
        entries, classes, length = x.size()
        # spread_output = torch.zeros((entries, classes, length + pad_left + pad_right))

        # spread_output and x should be in the same device (cuda)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        spread_output = torch.zeros((entries, classes, length + pad_left + pad_right), device=device)

        # print(spread_output.shape)
        # Fill the spread_output tensor
        for i in range(0, length):
            spread_output[:, :, i: i + self.kernel_size] += x[:, :, i:i + 1]
        # exit(0)

        spread_output = spread_output[:, :, pad_left:-pad_right]  # Trim padding

        return spread_output


import torch


def relu_max(input):
    # Find the maximum value along each row
    sigmoid_input = torch.sigmoid(input)
    max_values, _ = torch.max(sigmoid_input, dim=-1, keepdim=True)

    # Create a mask of the same shape as the tensor where 1 corresponds to the maximum value and 0 otherwise
    mask = torch.eq(sigmoid_input, max_values).to(input.device)

    # Set all values to -10000 except for the maximum value in each row
    output = torch.where(mask, input, torch.tensor(-10000.).to(input.device))
    return output


class ParallelCNNDecoders(nn.Module):
    def __init__(self, input_size, output_sizes, out_channels=8, kernel_size=14, droprate=0.3):
        super(ParallelCNNDecoders, self).__init__()
        self.cnn_decoders = nn.ModuleList([
            SimpleCNN(in_channels=input_size, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding='same', droprate=droprate) for output_size in output_sizes
        ])

    def forward(self, x):
        decoder_outputs = []
        motif_index = 0
        for decoder in self.cnn_decoders:
            outputs = decoder(x.permute(0, 2, 1))  # .squeeze(1) #[batch,maxlen-2]
            # if motif_index not in [0,4] :
            #   outputs = relu_max(outputs)

            decoder_outputs.append(outputs)
            motif_index += 1

        decoder_outputs = torch.stack(decoder_outputs, dim=1).squeeze(-1)  # [batch, num_class, maxlen-2]
        # decoder_outputs=torch.sigmoid(decoder_outputs)
        return decoder_outputs


class ParallelCNN_Linear_Decoders(nn.Module):
    def __init__(self, input_size, cnn_output_sizes, linear_output_sizes, out_channels=8, kernel_size=14, droprate=0.3):
        super(ParallelCNN_Linear_Decoders, self).__init__()
        self.num_decoders = len(cnn_output_sizes) + len(linear_output_sizes)
        self.linear_decoders = nn.ModuleList([
            nn.Linear(input_size, output_size) for output_size in linear_output_sizes
        ])

        self.cnn_decoders = nn.ModuleList([
            SimpleCNN(in_channels=input_size, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding='same', droprate=droprate) for output_size in cnn_output_sizes
        ])

    def forward(self, x):
        # decoder_outputs = [decoder(x) for decoder in self.linear_decoders]
        decoder_outputs = []
        for motif_index in range(self.num_decoders):
            if motif_index == 0:
                outputs = self.cnn_decoders[motif_index](x.permute(0, 2, 1)).squeeze(-1)
            elif motif_index >= 1 and motif_index <= 3:
                outputs = self.linear_decoders[motif_index - 1](x).squeeze(-1)  # sould be [batch,maxlen-2,1]
            elif motif_index == 4:
                outputs = self.cnn_decoders[1](x.permute(0, 2, 1)).squeeze(-1)
            elif motif_index >= 5:
                outputs = self.linear_decoders[motif_index - 2](x).squeeze(-1)  # sould be [batch,maxlen-2,1]

            decoder_outputs.append(outputs)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)  # [batch, num_class, maxlen-2]
        # decoder_outputs=torch.sigmoid(decoder_outputs)
        return decoder_outputs


class ParallelLinearDecoders(nn.Module):
    def __init__(self, input_size, output_sizes):
        super(ParallelLinearDecoders, self).__init__()
        self.linear_decoders = nn.ModuleList([
            nn.Linear(input_size, output_size) for output_size in output_sizes
        ])

    def forward(self, x):
        # decoder_outputs = [decoder(x) for decoder in self.linear_decoders]
        decoder_outputs = []
        motif_index = 0
        for decoder in self.linear_decoders:
            outputs = decoder(x).squeeze(-1)  # sould be [batch,maxlen-2,1]
            # if motif_index not in [0,4] :
            #   outputs = relu_max(outputs)
            decoder_outputs.append(outputs)
            motif_index += 1

        decoder_outputs = torch.stack(decoder_outputs, dim=1)  # [batch, num_class, maxlen-2]
        # decoder_outputs=torch.sigmoid(decoder_outputs)
        return decoder_outputs


def remove_s_e_token(target_tensor, mask_tensor):  # target_tensor [batch, seq+2, ...]  =>  [batch, seq, ...]
    # mask_tensor=inputs['attention_mask']
    # input_tensor=inputs['input_ids']
    result = []
    for i in range(mask_tensor.size()[0]):
        ind = torch.where(mask_tensor[i] == 0)[0]
        if ind.size()[0] == 0:
            result.append(target_tensor[i][1:-1])
        else:
            eos_ind = ind[0].item() - 1
            result.append(torch.concatenate((target_tensor[i][1:eos_ind], target_tensor[i][eos_ind + 1:]), axis=0))

    new_tensor = torch.stack(result, axis=0)
    return new_tensor


class Encoder(nn.Module):
    def __init__(self, configs, model_name='facebook/esm2_t33_650M_UR50D', model_type='esm_v2'):
        super().__init__()
        self.model_type = model_type
        if model_type == 'esm_v2':
            self.model = prepare_esm_model(model_name, configs)
        # self.pooling_layer = nn.AdaptiveAvgPool2d((None, 1))
        self.combine = configs.decoder.combine
        self.apply_DNN = configs.decoder.apply_DNN
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        if configs.decoder.type == "linear":
            self.ParallelDecoders = ParallelLinearDecoders(input_size=self.model.config.hidden_size,
                                                           output_sizes=[1] * configs.encoder.num_classes)
        elif configs.decoder.type == "cnn":
            self.ParallelDecoders = ParallelCNNDecoders(input_size=self.model.config.hidden_size,
                                                        output_sizes=[1] * configs.encoder.num_classes,
                                                        out_channels=configs.decoder.cnn_channel,
                                                        kernel_size=configs.decoder.cnn_kernel,
                                                        droprate=configs.decoder.droprate,
                                                        )
        elif configs.decoder.type == "cnn-linear":
            self.ParallelDecoders = ParallelCNN_Linear_Decoders(input_size=self.model.config.hidden_size,
                                                                cnn_output_sizes=[1] * 2,  # cnn
                                                                linear_output_sizes=[1] * (
                                                                            configs.encoder.num_classes - 2),
                                                                out_channels=configs.decoder.cnn_channel,
                                                                kernel_size=configs.decoder.cnn_kernel,
                                                                droprate=configs.decoder.droprate,
                                                                )
        if not self.combine:  # only need type_head if combine is False
            self.type_head = nn.Linear(self.model.embeddings.position_embeddings.embedding_dim,
                                       configs.encoder.num_classes)
        if self.combine == False and self.apply_DNN == True:
            print('combine and apply_DNN must be both True')
            raise 'combine and apply_DNN must be both True'
        if self.combine and self.apply_DNN:
            self.DNN_head = nn.Linear(configs.encoder.num_classes,
                                       configs.encoder.num_classes)

        self.overlap = configs.encoder.frag_overlap

        # SupCon
        self.apply_supcon = configs.supcon.apply
        if self.apply_supcon:
            self.projection_head = LayerNormNet(configs)
            self.n_pos = configs.supcon.n_pos
            self.n_neg = configs.supcon.n_neg
            self.batch_size = configs.train_settings.batch_size
            # new code

        self.skip_esm = configs.supcon.skip_esm
        if self.skip_esm:
            self.protein_embeddings = torch.load("5283_esm2_t33_650M_UR50D.pt")

        self.predict_max = configs.train_settings.predict_max

        # mini tools for supcon
        # curdir_path = os.getcwd()
        # tokenizer = prepare_tokenizer(configs, curdir_path)
        # self.tools = {
        # 'composition': configs.encoder.composition,
        # 'tokenizer': tokenizer,
        # 'max_len': configs.encoder.max_len,
        # 'train_device': configs.train_settings.device,
        # 'prm4prmpro': configs.encoder.prm4prmpro
        # }

        # self.mhatt = nn.MultiheadAttention(embed_dim=320, num_heads=10, batch_first=True)
        # self.attheadlist = []
        # self.headlist = []
        # for i in range(9):
        # self.attheadlist.append(nn.MultiheadAttention(embed_dim=320, num_heads=1, batch_first=True))
        # self.headlist.append(nn.Linear(320, 1))
        # self.device = device
        # self.device=configs.train_settings.device

    def get_pro_emb(self, id, id_frags_list, seq_frag_tuple, emb_frags, overlap):
        # print(seq_frag_tuple)
        # print('emb_frag', emb_frags.shape)
        emb_pro_list = []
        for id_protein in id:
            ind_frag = 0
            id_frag = id_protein + "@" + str(ind_frag)
            while id_frag in id_frags_list:
                ind = id_frags_list.index(id_frag)
                emb_frag = emb_frags[ind]  # [maxlen-2, dim]
                seq_frag = seq_frag_tuple[ind]
                l = len(seq_frag)
                if ind_frag == 0:
                    emb_pro = emb_frag[:l]
                else:
                    overlap_emb = (emb_pro[-overlap:] + emb_frag[:overlap]) / 2
                    emb_pro = torch.concatenate((emb_pro[:-overlap], overlap_emb, emb_frag[overlap:l]), axis=0)
                ind_frag += 1
                id_frag = id_protein + "@" + str(ind_frag)
            # print('-before mean', emb_pro.shape)
            emb_pro = torch.mean(emb_pro, dim=0)
            # print('-after mean', emb_pro.shape)
            emb_pro_list.append(emb_pro)

        emb_pro_list = torch.stack(emb_pro_list, dim=0)
        return emb_pro_list

    def get_pro_class(self, predict_max, id, id_frags_list, seq_frag_tuple, motif_logits, overlap):
        # motif_logits_max, _ = torch.max(motif_logits, dim=-1, keepdim=True).squeeze(-1) #should be [batch,num_class]
        # print(motif_logits_max)
        motif_pro_list = []
        motif_pro_list_dnn = []
        for id_protein in id:
            ind_frag = 0
            id_frag = id_protein + "@" + str(ind_frag)
            while id_frag in id_frags_list:
                ind = id_frags_list.index(id_frag)
                motif_logit = motif_logits[ind]  # [num_class,max_len]
                seq_frag = seq_frag_tuple[ind]
                l = len(seq_frag)
                if ind_frag == 0:
                    motif_pro = motif_logit[:, :l]  # [num_class,length]
                else:
                    overlap_motif = (motif_pro[:, -overlap:] + motif_logit[:, :overlap]) / 2
                    motif_pro = torch.concatenate((motif_pro[:, :-overlap], overlap_motif, motif_logit[:, overlap:l]),
                                                  axis=-1)
                ind_frag += 1
                id_frag = id_protein + "@" + str(ind_frag)

            if predict_max:
                # print('-before max', motif_pro.shape)  # should be [num_class,length]
                _motif_pro, _ = torch.max(motif_pro, dim=-1)
                # print('-after max', motif_pro.shape)  # should be [num_class]
                # print(motif_pro)
            else:
                # print('-before mean', motif_pro.shape)  # should be [num_class,length]
                # motif_pro = torch.mean(motif_pro, dim=-1)
                _motif_pro = torch.mean(motif_pro.clone(), dim=-1)  # yichuan 0605
                # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
                # 只在特定版本pytorch上出现此错误
                # print('-after mean', motif_pro.shape)  # should be [num_class]

            # motif_pro_list_dnn.append(motif_pro)
            motif_pro_list.append(_motif_pro)  # [batch,num_class]

        motif_pro_list = torch.stack(motif_pro_list, dim=0)
        # motif_pro_list_dnn = torch.stack(motif_pro_list_dnn, dim=0)
        # print('motif_pro_list', motif_pro_list.shape)
        # print('motif_pro_list_dnn', motif_pro_list_dnn)
        return motif_pro_list
    def get_pro_class_dnn(self, predict_max, id, id_frags_list, seq_frag_tuple, motif_logits, overlap):
        motif_pro_list = []
        motif_pro_list_dnn = []
        print(len(id), len(id_frags_list))
        for id_protein in id:
            ind_frag = 0
            id_frag = id_protein + "@" + str(ind_frag)
            while id_frag in id_frags_list:
                ind = id_frags_list.index(id_frag)
                motif_logit = motif_logits[ind]  # [num_class,max_len]
                seq_frag = seq_frag_tuple[ind]
                l = len(seq_frag)
                if ind_frag == 0:
                    motif_pro = motif_logit[:, :l]  # [num_class,length]
                else:
                    overlap_motif = (motif_pro[:, -overlap:] + motif_logit[:, :overlap]) / 2
                    motif_pro = torch.concatenate((motif_pro[:, :-overlap], overlap_motif, motif_logit[:, overlap:l]),
                                                  axis=-1)
                    print(motif_pro.shape)
                ind_frag += 1
                id_frag = id_protein + "@" + str(ind_frag)

            if predict_max:
                _motif_pro, _ = torch.max(motif_pro, dim=-1)
            else:
                _motif_pro = torch.mean(motif_pro.clone(), dim=-1)

            motif_pro_list.append(_motif_pro)  # [batch,num_class]

        motif_pro_list = torch.stack(motif_pro_list, dim=0)
        return motif_pro_list

    def reorganize_emb_pro(self, emb_pro):
        n_batch = int(emb_pro.shape[0] / (1 + self.n_pos + self.n_neg))
        bch_anchors, bch_positives, bch_negatives = torch.split(emb_pro,
                                                                [n_batch, n_batch * self.n_pos, n_batch * self.n_neg],
                                                                dim=0)
        emb_pro_ = []
        for i in range(n_batch):
            anchor = bch_anchors[i].unsqueeze(0)
            positive = bch_positives[(i * self.n_pos):(i * self.n_pos + self.n_pos)]
            negative = bch_negatives[(i * self.n_neg):(i * self.n_neg + self.n_neg)]
            triple = torch.cat((anchor, positive, negative), dim=0)
            emb_pro_.append(triple)
        emb_pro_ = torch.stack(emb_pro_, dim=0)
        return emb_pro_

    def forward(self, encoded_sequence, id, id_frags_list, seq_frag_tuple, pos_neg, warm_starting):
        """
        Batch is built before forward(), in train_loop()
        Batch is either (anchor) or (anchor+pos+neg)

        if apply supcon:
            if not warming starting:
                if pos_neg is None: ------------------> batch should be built as (anchor) in train_loop()
                    "CASE A"
                    get motif_logits from batch
                    get classification_head from batch
                else: --------------------------------> batch should be built as (anchor+pos+neg) in train_loop()
                    "CASE B"
                    get motif_logits from batch
                    get classification_head from batch
                    get projection_head from batch
            else: ------------------------------------> batch should be built as (anchor+pos+neg) in train_loop()
                "CASE C"
                get projection_head from batch
        else: ----------------------------------------> batch should be built as (anchor) in train_loop()
            "CASE D"
            get motif_logits from batch
            get classification_head from batch
        """
        classification_head = None
        motif_logits = None
        projection_head = None
        if self.skip_esm:
            pass
            # emb_pro_list = []
            # for i in id:
            #     emb_pro_list.append(self.protein_embeddings[i])
            # 
            # emb_pro = torch.stack(emb_pro_list, dim=0)
        else:
            # print(encoded_sequence['attention_mask'].shape)
            # print(encoded_sequence['attention_mask'])
            # print(encoded_sequence['input_ids'])
            # exit(0)
            # 这
            features = self.model(input_ids=encoded_sequence['input_ids'],
                                  attention_mask=encoded_sequence['attention_mask'])
            # print(features)
            last_hidden_state = remove_s_e_token(features.last_hidden_state,
                                                 encoded_sequence['attention_mask'])  # [batch, maxlen-2, dim]

            if not self.combine:  # only need type_head if combine is False
                emb_pro = self.get_pro_emb(id, id_frags_list, seq_frag_tuple, last_hidden_state, self.overlap)

        if self.apply_supcon:
            pass
            # if not warm_starting:
            #     motif_logits = self.ParallelDecoders(last_hidden_state)
            #     if self.combine:
            #         classification_head = self.get_pro_class(self.predict_max, id, id_frags_list, seq_frag_tuple,
            #                                                  motif_logits, self.overlap)
            #     else:
            #         classification_head = self.type_head(emb_pro)  # [sample, num_class]
            # 
            #     if pos_neg is not None:
            #         """CASE B, when this if condition is skipped, CASE A"""
            #         projection_head = self.projection_head(self.reorganize_emb_pro(emb_pro))
            # else:
            #     """CASE C"""
            #     projection_head = self.projection_head(self.reorganize_emb_pro(emb_pro))
        else:
            """CASE D"""
            """这"""
            motif_logits = self.ParallelDecoders(
                last_hidden_state)  # list no shape # last_hidden_state=[batch, maxlen-2, dim]
            if self.combine:
                classification_head = self.get_pro_class(self.predict_max, id, id_frags_list, seq_frag_tuple,
                                                         motif_logits, self.overlap)
                # print('classification_head', classification_head.shape)
                if self.apply_DNN:
                    classification_head = self.DNN_head(classification_head)
                    # print('classification_head', classification_head.shape)
                # if self.apply_DNN:
                #     classification_head = self.get_pro_class_dnn(self.predict_max, id, id_frags_list, seq_frag_tuple,
                #                                              motif_logits, self.overlap)
                # else:
                #     classification_head = self.get_pro_class(self.predict_max, id, id_frags_list, seq_frag_tuple,
                #                                              motif_logits, self.overlap)

            else:
                # print('emb_pro', emb_pro.shape)
                classification_head = self.type_head(emb_pro)  # [sample, num_class]
                # print('classification_head', classification_head.shape)

        # print(motif_logits[0,0,:])
        # print(motif_logits.shape)
        # print(motif_logits[0,1,:])
        # maxvalues,_ = torch.max(motif_logits[0], dim=-1, keepdim=True)
        # print(maxvalues)
        return classification_head, motif_logits, projection_head


class Bothmodels(nn.Module):
    def __init__(self, configs, pretrain_loc, trainable_layers, model_name='facebook/esm2_t33_650M_UR50D',
                 model_type='esm_v2'):
        super().__init__()
        self.model_esm = prepare_esm_model(model_name, configs)
        self.model_promprot = initialize_PromptProtein(pretrain_loc, trainable_layers)
        self.pooling_layer = nn.AdaptiveAvgPool1d(output_size=1)
        self.head = nn.Linear(self.model_esm.embeddings.position_embeddings.embedding_dim + 1280,
                              configs.encoder.num_classes)
        self.cs_head = nn.Linear(self.model_esm.embeddings.position_embeddings.embedding_dim, 1)

    def forward(self, encoded_sequence):
        features = self.model_esm(input_ids=encoded_sequence["encoded_sequence_esm2"]['input_ids'],
                                  attention_mask=encoded_sequence["encoded_sequence_esm2"]['attention_mask'])
        transposed_feature = features.last_hidden_state.transpose(1, 2)
        pooled_features_esm2 = self.pooling_layer(transposed_feature).squeeze(2)

        cs_head = self.cs_head(features.last_hidden_state).squeeze(dim=-1)
        cs_pred = cs_head[:, 1:201]

        features = self.model_promprot(encoded_sequence["encoded_sequence_promprot"], with_prompt_num=1)['logits']
        transposed_feature = features.transpose(1, 2)
        pooled_features_promprot = self.pooling_layer(transposed_feature).squeeze(2)

        pooled_features = torch.cat((pooled_features_esm2, pooled_features_promprot), dim=1)

        classification_head = self.head(pooled_features)

        return classification_head, cs_pred

def prepare_models(configs, logfilepath, curdir_path):
    if configs.encoder.composition == "esm_v2":
        encoder = Encoder(model_name=configs.encoder.model_name,
                          model_type=configs.encoder.model_type,
                          configs=configs
                          )
    if configs.encoder.composition=="official_esm_v2":
        encoder = OfficialEsmEncoder(model_name=configs.encoder.model_name,
                            model_type=configs.encoder.model_type,
                            configs=configs
                            )
    elif configs.encoder.composition == "promprot":
        encoder = CustomPromptModel(configs=configs,
                                    pretrain_loc=os.path.join(curdir_path, "PromptProtein", "PromptProtein.pt"),
                                    trainable_layers=["layers.32", "emb_layer_norm_after"])
    elif configs.encoder.composition == "both":
        encoder = Bothmodels(configs=configs,
                             pretrain_loc=os.path.join(curdir_path, "PromptProtein", "PromptProtein.pt"),
                             trainable_layers=[], model_name=configs.encoder.model_name,
                             model_type=configs.encoder.model_type)
    if not logfilepath == "":
        print_trainable_parameters(encoder, logfilepath)
    return encoder


class MaskedLMDataCollator:
    """Data collator for masked language modeling.

    The idea is based on the implementation of DataCollatorForLanguageModeling at
    https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L751C7-L782
    """

    def __init__(self, batch_converter, mlm_probability=0.15):
        """_summary_

        Args:
            mlm_probability (float, optional): The probability with which to (randomly) mask tokens in the input. Defaults to 0.15.
        """
        self.mlm_probability = mlm_probability
        """github ESM2
        self.special_token_indices = [batch_converter.alphabet.cls_idx, 
                                batch_converter.alphabet.padding_idx, 
                                batch_converter.alphabet.eos_idx,
                                #batch_converter.eos_token_id,
                                batch_converter.alphabet.unk_idx, 
                                batch_converter.alphabet.mask_idx,
                                #batch_converter.mask_token_id
                                ]
        """
        self.special_token_indices = batch_converter.all_special_ids
        # self.vocab_size = batch_converter.alphabet.all_toks.__len__()#github esm2
        self.vocab_size = batch_converter.all_tokens.__len__()
        # self.mask_idx = batch_converter.alphabet.mask_idx #github esm2
        self.mask_idx = batch_converter.mask_token_id  # huggingface

    def get_special_tokens_mask(self, tokens):
        return [1 if token in self.special_token_indices else 0 for token in tokens]

    def mask_tokens(self, batch_tokens):
        """make a masked input and label from batch_tokens.

        Args:
            batch_tokens (tensor): tensor of batch tokens
            batch_converter (tensor): batch converter from ESM-2.

        Returns:
            inputs: inputs with masked
            labels: labels for masked tokens
        """
        ## mask tokens
        inputs = batch_tokens.clone().to(batch_tokens.device)  # clone not work for huggingface tensor! don't know why!
        labels = batch_tokens.clone().to(batch_tokens.device)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = [self.get_special_tokens_mask(val) for val in labels]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # must remove this otherwise, inputs will be change as well
        # labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_idx

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size,
                                     labels.shape, dtype=torch.long).to(batch_tokens.device)

        # print(indices_random)
        # print(random_words)
        # print(inputs)

        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

"""
2024 06 11 adapter addition
"""
# from Jiang, ParallelLinearDecoders
class ParallelLinear(nn.Module):
    def __init__(self, input_size, output_sizes):
        super(ParallelLinear, self).__init__()
        self.linear_decoders = nn.ModuleList([
            nn.Linear(input_size, output_size) for output_size in output_sizes
        ])

    def forward(self, x):
        decoder_outputs = [decoder(x) for decoder in self.linear_decoders]
        return decoder_outputs

import torch.nn.functional as F
class ProteinCNN(nn.Module):
    def __init__(self, input_dim, num_filters, filter_size):
        super(ProteinCNN, self).__init__()
        # Convolutional layer
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=filter_size, padding='same')
        # Linear layer to transform the output to [batch, seq_length]
        self.linear = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = x.squeeze(-1)
        return x


import esm_utilities

class OfficialEsmEncoder(nn.Module):
    def __init__(self, configs, model_name='esm2_t33_650M_UR50D', model_type='esm_v2'):
        super().__init__()
        self.model_type = model_type
        if model_type == 'official_esm_v2':
            self.model = esm_utilities.load_model(
                model_architecture=configs.encoder.model_name,
                num_end_adapter_layers=configs.encoder.adapter_lr_num)

            for param in self.model.parameters():
                param.requires_grad = False

            if configs.PEFT == "Adapter":
                for name, param in self.model.named_parameters():
                    if "adapter_layer" in name:
                        param.requires_grad = True

            if configs.PEFT == "Adapter_PFT":
                for p in self.model.layers[configs.train_settings.fine_tune_lr:].parameters():
                    p.requires_grad = True
                for name, param in self.model.named_parameters():
                    if "adapter_layer" in name:
                        param.requires_grad = True

        # self.pooling_layer = nn.AdaptiveAvgPool2d((None, 1))
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.ParallelLinearDecoders = ParallelLinear(input_size=self.model.embed_dim,
                                                             output_sizes=[1] * (configs.encoder.num_classes - 2))
        self.CNN_Nuc = ProteinCNN(input_dim=self.model.embed_dim, num_filters=configs.encoder.cnn_filter_num,
                                  filter_size=configs.encoder.cnn_filter_size)
        self.CNN_Nuc_exp = ProteinCNN(input_dim=self.model.embed_dim, num_filters=configs.encoder.cnn_filter_num,
                                      filter_size=configs.encoder.cnn_filter_size)

        self.type_head = nn.Linear(self.model.embed_dim, configs.encoder.num_classes)
        self.overlap = configs.encoder.frag_overlap

        self.num_layers = self.model.num_layers

    def get_pro_emb(self, id, id_frags_list, seq_frag_tuple, emb_frags, overlap):
        emb_pro_list = []
        for id_protein in id:
            ind_frag = 0
            id_frag = id_protein + "@" + str(ind_frag)
            while id_frag in id_frags_list:
                ind = id_frags_list.index(id_frag)
                emb_frag = emb_frags[ind]  # [maxlen-2, dim]
                seq_frag = seq_frag_tuple[ind]
                l = len(seq_frag)
                if ind_frag == 0:
                    emb_pro = emb_frag[:l]
                else:
                    overlap_emb = (emb_pro[-overlap:] + emb_frag[:overlap]) / 2
                    emb_pro = torch.concatenate((emb_pro[:-overlap], overlap_emb, emb_frag[overlap:l]), axis=0)
                ind_frag += 1
                id_frag = id_protein + "@" + str(ind_frag)
            emb_pro = torch.mean(emb_pro, dim=0)
            emb_pro_list.append(emb_pro)
        return emb_pro_list

    def forward(self, encoded_sequence, id, id_frags_list, seq_frag_tuple, placeholder1, placeholder2):
        features = self.model(
            tokens=encoded_sequence,
            repr_layers=[self.num_layers])["representations"][self.num_layers]

        last_hidden_state = features[:, 1:-1]  # [batch, maxlen-2, dim]

        motif_logits_6clas = self.ParallelLinearDecoders(last_hidden_state)
        motif_logits_6clas = torch.stack(motif_logits_6clas, dim=1).squeeze(-1)  # [batch, num_class-2, maxlen-2]
        motif_logits_nuc = self.CNN_Nuc(last_hidden_state).unsqueeze(1)  # [batch, 1, maxlen-2]
        motif_logits_nuc_exp = self.CNN_Nuc_exp(last_hidden_state).unsqueeze(1)  # [batch, 1, maxlen-2]
        tensor_6clas_part1 = motif_logits_6clas[:, :4,
                             :]  # the first part of the 6 class tensor [batch, 3, maxlen-2], nuc_exp will be insert at index 4
        tensor_6clas_part2 = motif_logits_6clas[:, 4:, :]  # the second part of the 6 class tensor [batch, 3, maxlen-2]
        motif_logits = torch.cat((motif_logits_nuc, tensor_6clas_part1, motif_logits_nuc_exp, tensor_6clas_part2),
                                 dim=1)  # [batch, num_class, maxlen-2]

        emb_pro_list = self.get_pro_emb(id, id_frags_list, seq_frag_tuple, last_hidden_state, self.overlap)
        emb_pro = torch.stack(emb_pro_list, dim=0)  # [sample, dim]
        # transposed_feature = emb_pro.transpose(1, 2)
        # pooled_features = self.pooling_layer(transposed_feature).squeeze(2) #[sample, dim]
        classification_head = self.type_head(emb_pro)  # [sample, num_class]

        return classification_head, motif_logits, None

# if __name__ == '__main__':










