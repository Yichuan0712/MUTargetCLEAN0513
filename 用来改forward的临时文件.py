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
    else:
        features = self.model(input_ids=encoded_sequence['input_ids'],
                              attention_mask=encoded_sequence['attention_mask'])
        # print(features)
        last_hidden_state = remove_s_e_token(features.last_hidden_state,
                                             encoded_sequence['attention_mask'])  # [batch, maxlen-2, dim]

        if not self.combine:  # only need type_head if combine is False
            emb_pro = self.get_pro_emb(id, id_frags_list, seq_frag_tuple, last_hidden_state, self.overlap)

    if self.apply_supcon:
        pass
    else:
        """CASE D"""
        """这"""
        motif_logits = self.ParallelDecoders(
            last_hidden_state)  # list no shape # last_hidden_state=[batch, maxlen-2, dim]
        if self.combine:
            pass
        else:
            classification_head = self.type_head(emb_pro)  # [sample, num_class]

    return classification_head, motif_logits, projection_head