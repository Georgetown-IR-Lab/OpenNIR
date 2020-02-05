MODELSPACE=modelspace=wsdm2020_demo
BERT_MODEL_PARAMS="trainer.grad_acc_batch=1 valid_pred.batch_size=4 test_pred.batch_size=4"

python -m onir.bin.init_dataset $MODELSPACE config/antique
for ranker in config/trivial ranker=mz_knrm ranker=mz_conv_knrm ranker=knrm ranker=pacrr ranker=matchpyramid ranker=drmm ranker=duetl config/conv_knrm "config/vanilla_bert $BERT_MODEL_PARAMS"
do
	python -m onir.bin.pipeline vocab.source=glove vocab.variant=cc-42b-300d $MODELSPACE config/antique $ranker pipeline.test=true ranker.add_runscore=True
done

python -m onir.bin.extract_bert_weights $MODELSPACE pipeline.bert_weights=wsdm2020_demo config/antique pipeline.test=true config/vanilla_bert $BERT_MODEL_PARAMS ranker.add_runscore=True

for ranker in config/cedr/knrm config/cedr/pacrr config/cedr/drmm
do
	python -m onir.bin.pipeline $MODELSPACE config/antique $ranker $BERT_MODEL_PARAMS vocab.bert_weights=wsdm2020_demo pipeline.test=true ranker.add_runscore=True
done
