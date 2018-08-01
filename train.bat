@echo off

SET model_type=gated
SET nb_epoch=50
SET batch_size=2
SET dropout=True
SET pad=True
SET target_size=50,50,50
SET save_root=results\pixelcnn3d_gated_pad_dropout
SET load_data_root=data\lectin594

REM Less relevant ones 
SET nb_res_blocks=1
SET nb_filters_d=1024
SET continue_training=False
SET loss=categorical_crossentropy


python -m tests.train3d --model_type %model_type% --nb_epoch %nb_epoch% --batch_size %batch_size% --pad %pad% --loss %loss%^
 --target_size %target_size% --save_root %save_root% --load_data_root %load_data_root% --dropout %dropout% --nb_filters_d %nb_filters_d%^
 --nb_res_blocks %nb_res_blocks% --continue_training %continue_training%
 
 
pause 