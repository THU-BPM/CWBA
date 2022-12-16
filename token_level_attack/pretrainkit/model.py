import torch
import torch.nn as nn
# from transformers.modeling_bart import BartForConditionalGeneration
import logging
import os
import numpy as np

from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput


class ColumnPredictionModel(nn.Module):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, bert_name, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(bert_name)
        # self.bert = BertModel(config, add_pooling_layer=False)
        # for name, parameters in self.bert.named_parameters():
        #     # import ipdb; ipdb.set_trace()
        #     # print(name)
        #     if 'layer.0' not in name or 'layer.1' not in name or 'layer.2' not in name or 'layer.3' not in name or 'layer.4' not in name:
        #         parameters.requires_grad = False
        # import ipdb; ipdb.set_trace()
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, num_labels),
            # nn.ReLU(),
            # nn.Linear(20, num_labels)
        )
        # nn.init.normal_(self.classifier.weight.data, mean=0, std=0.02)
        # import ipdb;
        # ipdb.set_trace()
        # self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else True
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # import ipdb; ipdb.set_trace()

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# WEIGHTS_NAME = "pytorch_model.bin"
#
#
# class ColumnPredictionModel(nn.Module):
#   """
#   output: tuple: (loss, ) in training
#   """
#   def __init__(self, task):
#     super().__init__()
#     self.task = task
#     self.bert = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
#
#   def forward(self, *input, **kwargs):
#
#     input_ids = kwargs.pop("input_ids")
#
#     pad_token_id = kwargs.pop("pad_token_id")
#     begin_list = kwargs.pop("begin_list")
#     end_list = kwargs.pop("end_list")
#
#     label_eos_id = kwargs.pop("label_eos_id")
#     label_bos_id = kwargs.pop("label_bos_id")
#     column_labels = kwargs.pop("col_labels")
#     label_padding_id = kwargs.pop("label_padding_id")
#     attention_mask = (input_ids != pad_token_id).long()
#
#     if self.training:
#       output_ids = kwargs.pop('labels')
#       y_ids = output_ids[:, :-1].contiguous()
#       lm_labels = output_ids[:, 1:].clone()
#       lm_labels[output_ids[:, 1:] == pad_token_id] = -100
#
#       outputs = self.bert(input_ids,
#                           attention_mask=attention_mask, decoder_input_ids=y_ids, labels=lm_labels, output_hidden_states = True, return_dict=True)
#
#       if self.task == 'normal':
#         return (outputs['loss'],)
#       encoder_hidden_state = outputs['encoder_last_hidden_state']
#       decoder_hidden_state = outputs['decoder_hidden_states'][-1]
#       labels = lm_labels.cpu().numpy()
#       lengths = np.argwhere(labels==2)
#       criterion = nn.BCEWithLogitsLoss()
#       loss_list = []
#       for batch_idx in range(encoder_hidden_state.shape[0]):
#         enc_col_begin = encoder_hidden_state[batch_idx][begin_list[batch_idx]]
#         enc_col_end = encoder_hidden_state[batch_idx][end_list[batch_idx]]
#         enc_col = (enc_col_begin + enc_col_end)/2
#         attn_logits_list = []
#         for index in range(lengths[batch_idx][1] - 1):
#           query = decoder_hidden_state[batch_idx][index]
#           attn_logits = self.attention(query.unsqueeze(0), enc_col.unsqueeze(0))
#           attn_logits_list.append(attn_logits)
#         try:
#           attn_logits = torch.mean(torch.stack(attn_logits_list), dim=0)
#         except:
#           continue
#         col_label = column_labels[batch_idx][:attn_logits.shape[1]]
#         current_loss = criterion(attn_logits, col_label.unsqueeze(0))
#         loss_list.append(current_loss)
#       total_loss = torch.mean(torch.stack(loss_list), dim=0)
#       # import ipdb; ipdb.set_trace()
#       return (total_loss + outputs['loss'],)
#
#   def save_pretrained(self, save_directory):
#     """ Save a model and its configuration file to a directory, so that it
#         can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
#
#         Arguments:
#             save_directory: directory to which to save.
#     """
#     assert os.path.isdir(
#       save_directory
#     ), "Saving path should be a directory where the model and configuration can be saved"
#
#     # Only save the model itself if we are using distributed training
#     model_to_save = self.module if hasattr(self, "module") else self
#
#     # Attach architecture to the config
#     # model_to_save.config.architectures = [model_to_save.__class__.__name__]
#
#     # If we save using the predefined names, we can load using `from_pretrained`
#     output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
#
#     torch.save(model_to_save.state_dict(), output_model_file)




