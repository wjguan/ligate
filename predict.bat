@echo off

SET model_type=gated
SET batch_size=1
SET checkpoint_file=results\pixelcnn3d_gated_pad_dropout_logloss\pixelcnn-weights.50-35244689.1026.hdf5
SET pad=True
SET target_size=50,50,50
SET save_root=results\pixelcnn3d_gated_pad_dropout_logloss\test_images
SET load_data_file=data\lectin594\X_val.zarr
SET normalization_root=data\lectin594
REM Less relevant ones 
SET nb_res_blocks=1
SET nb_filters_d=1024
SET nb_images=4
SET temperature=1.0
SET loss=discretized_mix_logistic_loss

REM This is for predicting padded model 
python -m tests.predict3d --load_data_file %load_data_file% --model_type %model_type% --checkpoint_file %checkpoint_file%  --temperature %temperature%^
 --save_dir %save_root% --normalization_root %normalization_root% --target_size %target_size% --pad %pad% --nb_images %nb_images%^
 --batch_size %batch_size% --nb_res_blocks %nb_res_blocks% --nb_filters_d %nb_filters_d% --loss %loss% 
 
 
REM REM ####### 
REM REM running multiple predictions 
REM SET model_type=gated
REM SET checkpoint_file=results\pixelcnn3d_gated_pad_dropout_2resblocks_256filtersd\pixelcnn-weights.50-1.7228.hdf5
REM SET save_root=results\pixelcnn3d_gated_pad_dropout_2resblocks_256filtersd\test_images
REM SET nb_res_blocks=2
REM SET nb_filters_d=256 
REM REM This is for predicting padded model 
REM python -m tests.predict3d --load_data_file %load_data_file% --model_type %model_type% --checkpoint_file %checkpoint_file%  --temperature %temperature%^
 REM --save_dir %save_root% --normalization_root %normalization_root% --nb_images %nb_images% --target_size %target_size% --pad %pad%^
 REM --nb_res_blocks %nb_res_blocks% --nb_filters_d %nb_filters_d%
 
 
 
 
REM running multiple predictions 
REM SET model_type=vanilla
REM SET checkpoint_file=results\pixelcnn3d_pad\pixelcnn-weights.50-1.9193.hdf5
REM SET save_root=results\pixelcnn3d_pad\test_images
REM SET nb_res_blocks=1
REM SET nb_filters_d=1024 
REM SET pad=True
REM SET target_size=50,50,50
REM REM This is for predicting padded model 
REM python -m tests.predict3d --model_type %model_type% --checkpoint_file %checkpoint_file% --load_data_file %load_data_file% --temperature %temperature%^
 REM --save_dir %save_root% --normalization_root %normalization_root% --nb_images %nb_images% --target_size %target_size% --pad %pad% --nb_images %nb_images%

pause 