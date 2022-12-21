from transformers import BertTokenizerFast, EncoderDecoderModel, BertModel, RobertaModel, AutoConfig, BertConfig, RobertaConfig, TrainingArguments, Seq2SeqTrainer
from torchinfo import summary


out_f = open('b2b_config_params.csv', 'w')
print('hidden_dim,n_layer_enc,n_layer_dec,n_params', file=out_f)


for hidden in [128, 256, 512, 768]:
    for enc_layer in [2,4,6,8,10,12]:
        for dec_layer in [2,4,6,8,10,12]:
            encoder_name ='google/bert_uncased_L-{}_H-{}_A-{}'.format(
                enc_layer, hidden, hidden // 64)
            decoder_name ='google/bert_uncased_L-{}_H-{}_A-{}'.format(
                dec_layer, hidden, hidden // 64)
            bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_name, decoder_name)
            res = summary(bert2bert, verbose=0)
            print('{},{},{},{}'.format(hidden, enc_layer, dec_layer, res.total_params))
            print('{},{},{},{}'.format(hidden, enc_layer, dec_layer, res.total_params), 
                  file=out_f, flush=True)
