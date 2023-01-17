import tensorflow as tf
import os
import time

INT8_QUANTIZED = 1

FLOAT16_QUANTIZED = 0

INT8_OUTPUT = 0

UINT8_OUTPUT = 0

# FLOAT16_OUTPUT = 1 #not available

@tf.function
def parse_tfrecord_fn(serialized_example):

    feature={
      'image/height': tf.io.FixedLenFeature([], tf.int64),
      'image/width': tf.io.FixedLenFeature([], tf.int64),
      'image/filename': tf.io.FixedLenFeature([], tf.string),
      'image/source_id': tf.io.FixedLenFeature([], tf.string),
      'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
      'image/encoded': tf.io.FixedLenFeature([], tf.string),
      'image/format': tf.io.FixedLenFeature([], tf.string),
      'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
      'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
      'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
      'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
      'image/object/class/text': tf.io.VarLenFeature(tf.string),
      'image/object/class/label': tf.io.VarLenFeature(tf.int64),
      'image/object/difficult': tf.io.VarLenFeature(tf.int64),
      'image/object/track/label': tf.io.VarLenFeature(tf.int64),
    }

    serialized_example = tf.io.parse_single_example(serialized_example, feature)
    
    image = tf.io.decode_jpeg(serialized_example['image/encoded'])
    
    return image



def representative_dataset():
    record_file_name = "/workspace/dataset/caltech/val.tfrecord"
    image_index = 0
    for data in tf.data.TFRecordDataset(record_file_name, num_parallel_reads=1).map(parse_tfrecord_fn, num_parallel_calls=2).take(200):
        image = tf.image.resize(data, (384, 384))
        # image = tf.cast(image / 255., tf.float32)
        image_index =  image_index + 1
        print(f'sampled image index for quantization is {image_index}')
        #image = tf.cast((image - 127.5)/128., tf.float32)
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, 0)
        if (image.shape[3]!=3):
            continue
        yield[image]
        

if __name__ == "__main__":   
    converter = tf.lite.TFLiteConverter.from_saved_model("/workspace/MultiObjTracking/pb/saved_model/")
        
    converter.allow_custom_ops=True
    # converter._quantize = True

    # Convert the model to quantized TFLite model.
    converter.optimizations =  [tf.lite.Optimize.DEFAULT]
    
    # converter.optimizations =  [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    
    converter.experimental_new_converter = True
    
    if INT8_QUANTIZED:
        converter.representative_dataset =  representative_dataset
    
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                            tf.lite.OpsSet.TFLITE_BUILTINS,
                                            tf.lite.OpsSet.SELECT_TF_OPS]
    elif FLOAT16_QUANTIZED:
        converter.representative_dataset =  representative_dataset
    
        converter.target_spec.supported_ops = [tf.float16,
                                            tf.lite.OpsSet.TFLITE_BUILTINS,
                                            tf.lite.OpsSet.SELECT_TF_OPS]
    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                    tf.lite.OpsSet.SELECT_TF_OPS]
    

    if INT8_OUTPUT:
        converter.inference_output_type = tf.int8
    elif UINT8_OUTPUT:
        converter.inference_output_type = tf.uint8
    # elif FLOAT16_OUTPUT: #not available
    #     converter.inference_output_type = tf.float16
    else:
        converter.inference_output_type = tf.float32
    
    print(os.getpid())
    
    tflite_model = converter.convert()
    
    now = time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime(time.time())) 
    
    filename = '/workspace/MultiObjTracking/tflites/TFLite_Tracking_'+('quantized_int8_' if INT8_QUANTIZED else ('quantized_float16_' if FLOAT16_QUANTIZED else 'non_quantized_')) \
        + ('int8_output_' if INT8_OUTPUT else ('uint8_output_'if UINT8_OUTPUT else 'float_output_'))+now+'.tflite'

    # Write a model using the following line
    open(filename, "wb").write(tflite_model)
     
    pass
