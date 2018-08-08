@echo off

SET model_type=gated
SET batch_size=1
SET checkpoint_file=results\bidir_gated_pad_dropout_2resblocks_weighted\pixelcnn-weights.34-1.7741.hdf5
SET pad=True
SET target_size=50,50,50
SET save_root=results\bidir_gated_pad\test_images
SET load_data_file=data\lectin594\X_val.zarr
SET normalization_root=data\lectin594
REM Less relevant ones 
SET nb_res_blocks=2
SET nb_filters=100
SET nb_images=4
SET temperature=1.0
SET loss=weighted_categorical_crossentropy
SET dropout=True

REM This is for predicting padded model 
python -m tests.predict_bigate --load_data_file %load_data_file% --model_type %model_type% --checkpoint_file %checkpoint_file%  --temperature %temperature%^
 --save_dir %save_root% --normalization_root %normalization_root% --target_size %target_size% --pad %pad% --nb_images %nb_images%^
 --batch_size %batch_size% --nb_res_blocks %nb_res_blocks% --nb_filters %nb_filters% --loss %loss% --dropout %dropout%

 pause