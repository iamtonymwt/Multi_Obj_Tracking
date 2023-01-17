import os
import subprocess
import argparse
import re
import json
#import demjson
import sys

NON_NORMALIZATION_INPUT = 0

QUANTIZED_OUTPUT = 1

All_PostProcess = 0

Argmax_PostProcess = 0

ObArgmax_PostProcess = 0

Row_NonPostProcess = 0

Heatmap_NonPostProcess = 1

OC_RG_NonPostProcess = 0

removed_lines_list = []

if NON_NORMALIZATION_INPUT:
    removed_node_list = []
else:
    if QUANTIZED_OUTPUT:
        removed_node_list = [1,2]
    else:
        removed_node_list = [0,1] # for non quantized weight AI model

# node index starts from 0
def operators_remove(linelist, removed_node_list):
    node_count = 0
    operators_range_start = False
    node_count_start = False
    node_end = False
    last_output_index = None
    first_replaced_index = None
    replacement_start = False
    remove_count_start = False
    
    if len(removed_node_list) > 0:
        for line_index, line in enumerate(linelist):
            if "operators: [" in line:
                operators_range_start = True
            
            if "{" in line and operators_range_start:
                node_count_start = True
            elif "}," in line and node_count_start:
                node_count_start = False
                node_count = node_count + 1
                node_end = True
      
        
            for i in removed_node_list:
                # if (node_count_start and (node_count == removed_node_list[i-1])) or (node_count == removed_node_list[i-1] + 1 and node_end):
                if (node_count_start and (node_count == i)) or (node_count == i + 1 and node_end):
                    remove_count_start = True
                    removed_lines_list.append(line_index)
                    if node_end:
                        node_end = False
        
            if node_count > max(removed_node_list):
                operators_range_start = False
                remove_count_start = False
                replacement_start = True
                node_count = 0
             
            #only one output condition
            if ("outputs: [" in line and operators_range_start and not remove_count_start) \
            or (remove_count_start and node_count == 0 and ("inputs: [" in line)):
                last_output_index = line_index + 1
            if "inputs: [" in line and replacement_start:
                first_replaced_index = line_index + 1
                replacement_start = False
            
        if "," not in linelist[first_replaced_index]:
            linelist[first_replaced_index] = linelist[last_output_index].rstrip() + "\n"
        else:
            linelist[first_replaced_index] = linelist[last_output_index].rstrip() + ",\n"       
    
    return removed_lines_list

if __name__ =='__main__':
    
    print('[FP] flatbuffer postprocess start!')
    
    parser = argparse.ArgumentParser(description="flatbuffer file postprocess parser")
    parser.add_argument('-tfp', dest='filepath', type=str, required=True, help='tflite file path')
    args = parser.parse_args()
    
    tflite_filename = (args.filepath).split(".")[0]
    
    print(tflite_filename)
    
    # os.chdir('tflites/')
    print(subprocess.getoutput('pwd'))
    print(subprocess.getoutput("flatc -t schema.fbs -- " +  tflite_filename + ".tflite"))
    #print(subprocess.getoutput("mv " + tflite_filename + ".json" + " flatbuffer/"))
    #print(subprocess.getoutput("ls -t flatbuffer/"))
    
    modified_json_file = tflite_filename + "_modified.json"
    print(subprocess.getoutput("cp "  + tflite_filename + ".json" + " " + modified_json_file))
    #print(subprocess.getoutput("ls -t tflites/"))
    
    print('[FP] ==> modified json file is prepared!')

    #For CenterNet Postprocess
    # if All_PostProcess:
    #     #Replacement
    #     search_list = ['deprecated_builtin_code: 88', 'UNPACK', 'UnpackOptions', 'serving_default_input:0', 
    #                 'StatefulPartitionedCall:01', 'StatefulPartitionedCall:0', 'StatefulPartitionedCall:11', 'StatefulPartitionedCall:1', 
    #                 'StatefulPartitionedCall:21', 'StatefulPartitionedCall:2',
    #                 'StatefulPartitionedCall:3',
    #                 'StatefulPartitionedCall:4']
    #     replace_list = ['deprecated_builtin_code: 43', 'SQUEEZE', 'SqueezeOptions', 'input', 
    #                     'keypoints_heatmap_scores_before_dq', 'keypoints_heatmap_scores', 'keypoints_regression_before_bq', 'keypoints_regression', 
    #                     'keypoints_heatmap_offsets_before_bq', 'keypoints_heatmap_offsets',
    #                     'object_center_argmax_indices',
    #                     'keypoints_heatmap_argmax_indices']
    
    # elif Argmax_PostProcess:
    #     # for argmax output version network
    #     search_list = ['deprecated_builtin_code: 88', 'UNPACK', 'UnpackOptions', 'serving_default_input:0', 
    #                 'StatefulPartitionedCall:7','StatefulPartitionedCall:6',  
    #                 'StatefulPartitionedCall:5','StatefulPartitionedCall:4',
    #                 'StatefulPartitionedCall:3','StatefulPartitionedCall:2',
    #                 'StatefulPartitionedCall:19','StatefulPartitionedCall:18',
    #                 'StatefulPartitionedCall:17','StatefulPartitionedCall:16',
    #                 'StatefulPartitionedCall:15','StatefulPartitionedCall:14',
    #                 'StatefulPartitionedCall:13','StatefulPartitionedCall:12',
    #                 'StatefulPartitionedCall:1','StatefulPartitionedCall:0',
    #                 'StatefulPartitionedCall:101',
    #                 'StatefulPartitionedCall:10',
    #                 'StatefulPartitionedCall:91',
    #                 'StatefulPartitionedCall:9',
    #                 'StatefulPartitionedCall:81',
    #                 'StatefulPartitionedCall:8',
    #                 'StatefulPartitionedCall:11']
    #     replace_list = ['deprecated_builtin_code: 43', 'SQUEEZE', 'SqueezeOptions', 'input', 
    #                     'keypoints_heatmap_argmax_indice1','keypoints_heatmap_argmax_indice2', 
    #                     'keypoints_heatmap_argmax_indice3','keypoints_heatmap_argmax_indice4',
    #                     'keypoints_heatmap_argmax_indice5','keypoints_heatmap_argmax_indice6',
    #                     'keypoints_heatmap_argmax_indice7','keypoints_heatmap_argmax_indice8',
    #                     'keypoints_heatmap_argmax_indice9','keypoints_heatmap_argmax_indice10',
    #                     'keypoints_heatmap_argmax_indice11','keypoints_heatmap_argmax_indice12',
    #                     'keypoints_heatmap_argmax_indice13','keypoints_heatmap_argmax_indice14',
    #                     'keypoints_heatmap_argmax_indice15','keypoints_heatmap_argmax_indice16',
    #                     'keypoints_heatmap_offsets_before_bq',
    #                     'keypoints_heatmap_offsets',
    #                     'keypoints_regression_before_bq',
    #                     'keypoints_regression',
    #                     'keypoints_heatmap_scores_before_dq', 
    #                     'keypoints_heatmap_scores',
    #                     'object_center_argmax_indices']
    
    # elif ObArgmax_PostProcess:
    #     # for only ob   ject argmax output version network
    #     search_list = ['deprecated_builtin_code: 88', 'UNPACK', 'UnpackOptions', 'serving_default_input:0', 
    #                 'StatefulPartitionedCall:3',
    #                 'StatefulPartitionedCall:1', 'StatefulPartitionedCall:11',
    #                 'StatefulPartitionedCall:0','StatefulPartitionedCall:01',
    #                 'StatefulPartitionedCall:2','StatefulPartitionedCall:21']
    #     replace_list = ['deprecated_builtin_code: 43', 'SQUEEZE', 'SqueezeOptions', 'input', 
    #                     'object_center_argmax_indices',
    #                     'keypoints_regression', 'keypoints_regression_before_bq',
    #                     'keypoints_heatmap','keypoints_heatmap_before_dq',
    #                     'keypoints_heatmap_offsets', 'keypoints_heatmap_offsets_before_bq']
        
    # elif Row_NonPostProcess:
    #     # for raw output version network
    #     search_list = ['deprecated_builtin_code: 88', 'UNPACK', 'UnpackOptions', 'serving_default_input:0', 
    #                 'StatefulPartitionedCall:3','StatefulPartitionedCall:31',
    #                 'StatefulPartitionedCall:1','StatefulPartitionedCall:11',
    #                 'StatefulPartitionedCall:0','StatefulPartitionedCall:01',
    #                 'StatefulPartitionedCall:2','StatefulPartitionedCall:21']
    #     replace_list = ['deprecated_builtin_code: 43', 'SQUEEZE', 'SqueezeOptions', 'input', 
    #                     'object_center_heatmap', 'object_center_heatmap_before_dq',
    #                     'keypoints_regression','keypoints_regression_before_bq', 
    #                     'keypoints_heatmap','keypoints_heatmap_before_dq',
    #                     'keypoints_heatmap_offsets','keypoints_heatmap_offsets_before_bq']
    # elif Heatmap_NonPostProcess:
    #     # for only heatmap output version network
    #     search_list = ['deprecated_builtin_code: 88', 'UNPACK', 'UnpackOptions', 'serving_default_input:0', 
    #                 'StatefulPartitionedCall:0','StatefulPartitionedCall:01',
    #                 'StatefulPartitionedCall:1','StatefulPartitionedCall:11']
    #     replace_list = ['deprecated_builtin_code: 43', 'SQUEEZE', 'SqueezeOptions', 'input',
    #                     'keypoints_heatmap','keypoints_heatmap_before_dq',
    #                     'keypoints_heatmap_offsets','keypoints_heatmap_offsets_before_bq']
    # elif OC_RG_NonPostProcess:
    #     # for only object center and regression version network
    #     search_list = ['deprecated_builtin_code: 88', 'UNPACK', 'UnpackOptions', 'serving_default_input:0', 
    #                 'StatefulPartitionedCall:0','StatefulPartitionedCall:01',
    #                 'StatefulPartitionedCall:1','StatefulPartitionedCall:11']
    #     replace_list = ['deprecated_builtin_code: 43', 'SQUEEZE', 'SqueezeOptions', 'input',
    #                     'object_center_heatmap', 'object_center_heatmap_before_dq',
    #                     'keypoints_regression','keypoints_regression_before_bq']
    
    #for only heatmap output version network
    search_list = ['UNPACK', 'UnpackOptions', 'serving_default_input:0', 
                'StatefulPartitionedCall:21','StatefulPartitionedCall:2',
                'StatefulPartitionedCall:11','StatefulPartitionedCall:1',
                'StatefulPartitionedCall:01','StatefulPartitionedCall:0',
                'StatefulPartitionedCall:31','StatefulPartitionedCall:3']
    replace_list = ['SQUEEZE', 'SqueezeOptions', 'input',
                    'boxes_offset_before_deq','boxes_offset',
                    'boxes_wh_before_deq','boxes_wh',
                    'obj_center_heatmap_before_deq','obj_center_heatmap',
                    'reid_embedding_before_deq','reid_embedding']


    with open(modified_json_file, 'r+') as f:
        file = f.read()
        
        for search_text, replace_text in zip(search_list, replace_list):
            file = re.sub("\\b"+search_text+"\\b", replace_text, file)
            pass
            # file = re.sub(search_text, replace_text, file)
        
        f.seek(0)
        f.write(file)
        f.truncate()
        
    print('[FP] ==> Identical OPs are replaced!')
    
    #Modify json value
    # modified_json_file_bak = "tflites/" + tflite_filename + "_modified.json.bak"
    # print(subprocess.getoutput("cp "+modified_json_file+" "+modified_json_file_bak))
    
    with open(modified_json_file, 'r') as jr:
        linelist = jr.readlines()
        search_s_bool = False
        for i, line in enumerate(linelist):
            if ("name: \"Sigmoid\"" in line) or ("name: \"Sigmoid_1\"" in line):
                search_s_bool = True
            if "0.003906" in line and search_s_bool:
                linelist[i] = line.replace("0.003906","0.00390625")
                search_s_bool = False
    
    print('[FP] ==> Sigmoid value is corrected!')
    
                
    with open(modified_json_file, 'w') as jw:
        node_remove_list = operators_remove(linelist,removed_node_list)
        for line_index, line in enumerate(linelist):
            if "builtin_options_type: \"SqueezeOptions\"" in line:
                search_s_bool = True
            if "num: 1" in line and search_s_bool:
                search_s_bool = False
                continue
            if line_index in node_remove_list:
                continue                
            jw.write(line)
        jw.truncate()
    print('[FP] ==> num line in SqueezeOptions and indicated nodes are removed!')

    print(subprocess.getoutput("flatc -b schema.fbs "+modified_json_file))
    # print(subprocess.getoutput("mv " + tflite_filename + "_modified.tflite" + " " + "tflites/"))
    print('[FP] ==> Modified tflite file is generated!')
    print('[FP] flatbuffer postprocess is done!')
