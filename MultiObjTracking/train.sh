
# python /workspace/google_api/tf2/research/object_detection/model_main_tf2.py \
#        --model_dir=/workspace/MultiObjTracking/output \
#        --num_train_steps=250000 \
#        --pipeline_config_path=/workspace/MultiObjTracking/mot.config \
#        --num_workers=20 \
#        ---checkpoint_every_n=1\
#        --alsologtostderr


python /workspace/google_api/tf2/research/object_detection/model_main_tf2.py \
       --model_dir=/workspace/MultiObjTracking/output \
       --num_train_steps=25000 \
       --pipeline_config_path=/workspace/MultiObjTracking/mot_qat.config \
       --num_workers=20 \
       ---checkpoint_every_n=1\
       --alsologtostderr
