# python /workspace/google_api/tf2/research/object_detection/export_tflite_graph_tf2.py \
#        --pipeline_config_path /workspace/MultiObjTracking/mot.config \
#        --trained_checkpoint_dir /workspace/MultiObjTracking/output \
#        --output_directory /workspace/MultiObjTracking/pb

python /workspace/google_api/tf2/research/object_detection/export_tflite_graph_tf2.py \
       --pipeline_config_path /workspace/MultiObjTracking/mot_qat.config \
       --trained_checkpoint_dir /workspace/MultiObjTracking/output \
       --output_directory /workspace/MultiObjTracking/pb
