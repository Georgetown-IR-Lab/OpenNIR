from pytorch_transformers.modeling_bert import BertForPreTraining, BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPreTrainingHeads


class CustomBertModelWrapper(BertForPreTraining):
    def __init__(self, config, depth=None):
        config.output_hidden_states = True
        super().__init__(config)
        self.bert = CustomBertModel(config, depth) # replace with custom model

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.bert(input_ids, token_type_ids, attention_mask)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        result = super().from_pretrained(*args, **kwargs)
        if result.bert.depth is not None:
            # limit the depth by cutting out layers it doesn't need to calculate
            result.bert.encoder.layer = result.bert.encoder.layer[:result.bert.depth]
        else:
            result.depth = len(result.bert.encoder.layer)
        return result

    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = trainable


class CustomBertModel(BertPreTrainedModel):
    """
    Based on pytorch_pretrained_bert.BertModel, but with some extra goodies:
     - depth: number of layers to run in BERT, where 0 is the raw embeddings, and -1 is all
              available layers
    """
    def __init__(self, config, depth=None):
        super(CustomBertModel, self).__init__(config)
        self.depth = depth
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertPreTrainingHeads(config)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        Based on pytorch_pretrained_bert.BertModel
        """
        embedding_output = self.embeddings(input_ids, token_type_ids)
        if self.depth == 0:
            return [embedding_output]

        return self.forward_from_layer(embedding_output, attention_mask)

    def forward_from_layer(self, embedding_output, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers

        _, encoded_layers = self.encoder(embedding_output, extended_attention_mask, head_mask)
        return list(encoded_layers)
