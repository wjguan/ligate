@echo off

SET model_type=gated
SET nb_epoch=50
SET batch_size=2
SET dropout=True
SET pad=True
SET target_size=32,32,32
SET save_root=results\bidir_32x32x32_gated_pad_dropout_5resblocks
SET load_data_root=data\lectin594_small
SET min_mask_size=4,4,4
SET max_mask_size=7,7,7

REM Less relevant ones 
SET nb_res_blocks=5
SET nb_filters=100
SET continue_training=False
SET loss=categorical_crossentropy
SET random_rotation=True
SET random_flip=False

python -m tests.train_bigate --model_type %model_type% --nb_epoch %nb_epoch% --batch_size %batch_size% --pad %pad% --loss %loss%^
 --target_size %target_size% --save_root %save_root% --load_data_root %load_data_root% --dropout %dropout% --nb_filters %nb_filters%^
 --nb_res_blocks %nb_res_blocks% --continue_training %continue_training% --min_mask_size %min_mask_size% --max_mask_size %max_mask_size%^
 --random_rotation %random_rotation% --random_flip %random_flip%
 
 
pause 