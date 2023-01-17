import tensorflow as tf
import cv2
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import seaborn as sns
import os
import time
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
import heapq
import math
import sys

QUANTIZED_OUTPUT = 0

#43
# ZeroPointValue = 47.3333
HFZeroPointValue = 43
#0.0660751610994339
HFQuantizedScale = 0.06607516

HMZeroPointValue = 118
# 0.2295181304216385
HMQuantizedScale = 0.22951813

OCZeroPointValue = -128
#0.00390625
OCQuantizedScale = 0.00390625

HRZeroPointValue = -55
#0.24907918274402618
HRQuantizedScale = 0.24907918

BHWZeroPointValue = 0

BHWQuantizedScale = 0

BOFZeroPointValue = 0

BOFQuantizedScale = 0

DISPLAY_HEATMAP = 0

SAVE_HEATMAP = 0
# SAVE_HEATMAP = 1

ONLY_HEATMAP_OUTPUT = 0

# HEATMAP_SIMPLIFIED_DEBUG = 1
HEATMAP_SIMPLIFIED_DEBUG = 0

HEATMAP_SIMPLIFIED_DEBUG_LAYER_INDEX = 232

OC_SIMPLIFIED_DEBUG_LAYER_INDEX = 221

ONLY_OC_RG_OUTPUT = 0

OBJECTC_HEATMAP_OUTPUT = 0

OBJECTC_WB_HEATMAP_OUTPUT = 0

OBJECT_WB_HEATMAP_REID_OUTPUT = 1

NORMALIZATION_INPUT = 0
# NORMALIZATION_INPUT = 1

# TOPK_SELECTION = 20
TOPK_SELECTION = 6
MAX_PEOPLE = 2

DEBUG_POINTS_INFO = 0

OC_RG = 0

if ONLY_OC_RG_OUTPUT:
    OC_RG = 1

LINE_VISIBILITY = 1

centerpoints_threshold_value = 0.1
# centerpoints_threshold_value = 0.12

# object_reID_eucDistance_threshold_value = 6
object_reID_eucDistance_threshold_value = 2
ID_TENSOR_NORMALIZE = False
DYNAMIC_UPDATE_ID_DIC = False
VIDEO_FRAME_FREQUENCY = 2

diff_exchagable_range = 1

distance_threshold = 1
 
multi_distance_scale = 3  
# matplotlib.use("Qt5Agg")

reIDDict = dict()

def heatmap_decode():
    pass

def heatmap_decode_multi_person():
    pass

def heatmap_decode_with_bbox():
    pass

def swap_scale_points(point, Hscale=16, Wscale=16):
    point_strided_y, point_strided_x = [i for i in point]
    # point_strided = point_strided_x, point_strided_y
    point_strided_int = int(point_strided_x*Wscale), int(point_strided_y*Hscale)
    return point_strided_int
    
    
def heatmap_save(image_name, object_center_heatmap, boxes_offsets, boxes_wh):
    centernet_heatmap_figs={}
    
    now = time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime(time.time())) 
    
    image_folder_path = 'heatmap_images/'+image_name+"_"+now
    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)
    
    output_format = "d" if QUANTIZED_OUTPUT else ".2f"
    
    if object_center_heatmap is not None:
        fig=plt.figure()
        object_center_heatmap_plot=sns.heatmap(object_center_heatmap[:,:,0], annot= True, fmt=output_format)
        object_center_heatmap_plot.set_title('object_center_heatmap')
        # fig.canvas.manager.window.setWindowTitle("object_center_heatmap")
        # plt.legend()
        fig.set_size_inches(50, 40)
        plt.grid(True) 
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        plt.savefig(image_folder_path+'/'+image_name+'_'+'object_center_heatmap'+'.jpg',dpi=100)
        plt.close(fig)
    
    if boxes_wh is not None:
        for index in range(2):
            fig=plt.figure()
            centernet_heatmap_figs[index] = sns.heatmap(boxes_wh[:,:,index], annot= True, fmt=output_format)    
            centernet_heatmap_figs[index].set_title('boxes_wh_point'+str(index))
            # fig.canvas.manager.window.setWindowTitle('boxes_wh_point'+str(index))
            # plt.legend()
            fig.set_size_inches(50, 40)
            plt.grid(True)
            # manager = plt.get_current_fig_manager()
            # manager.window.showMaximized()
            plt.savefig(image_folder_path+'/'+image_name+'_'+'boxes_wh_point'+str(index)+'.jpg',dpi=100)
            plt.close(fig)
            
    if boxes_offsets is not None:
        for index in range(2):
            fig=plt.figure()
            centernet_heatmap_figs[index] = sns.heatmap(boxes_offsets[:,:,index], annot= True, fmt=output_format)    
            centernet_heatmap_figs[index].set_title('boxes_offsets_point'+str(index))
            # fig.canvas.manager.window.setWindowTitle('boxes_offsets_point'+str(index))
            # plt.legend()
            fig.set_size_inches(50, 40)
            plt.grid(True)
            # manager = plt.get_current_fig_manager()
            # manager.window.showMaximized()
            plt.savefig(image_folder_path+'/'+image_name+'_'+'boxes_offsets_point'+str(index)+'.jpg',dpi=100)
            plt.close(fig)
        
    sys.exit(1)

def get_row_col_indices_from_flattened_indices(indices, num_cols_per_row=24):
    row_indices = np.array(indices) // num_cols_per_row
    col_indices = indices - row_indices * num_cols_per_row
    return row_indices, col_indices

# as only test heatmap related two branches, no utilization of topk currently   
def get_object_center_coordinates_scores(object_center_heatmap, topk=1):
    object_center_heatmap_flattened = object_center_heatmap.flatten()
    # object_center_heatmap_flattened_indices = heapq.nlargest(topk, range(len(object_center_heatmap_flattened)), object_center_heatmap_flattened.__getitem__)
    object_center_heatmap_flattened_indices = np.argsort(object_center_heatmap_flattened)[::-1][:topk]
    # object_center_heatmap_flattend_values = heapq.nlargest(topk, object_center_heatmap_flattened)
    object_center_heatmap_flattend_values = object_center_heatmap_flattened[object_center_heatmap_flattened_indices]
    
    object_center_y_indices, object_center_x_indices = get_row_col_indices_from_flattened_indices(object_center_heatmap_flattened_indices, object_center_heatmap.shape[1])
    
    # object_center_coordinates = np.unravel_index(np.argmax(object_center_heatmap, axis=None),object_center_heatmap.shape)
    # object_center_y,object_center_x, _ = object_center_coordinates
    # object_center_scores = ((object_center_heatmap[object_center_y][object_center_x][0] - OCZeroPointValue) * OCQuantizedScale) \
    #     if QUANTIZED_OUTPUT else object_center_heatmap[object_center_y][object_center_x][0]
    
    object_center_scores = ((object_center_heatmap_flattend_values - OCZeroPointValue) * OCQuantizedScale) \
        if QUANTIZED_OUTPUT else object_center_heatmap_flattend_values
        
    # return object_center_y,object_center_x, object_center_scores
    return object_center_y_indices, object_center_x_indices, object_center_scores

def cal_square_distance(p_y, p_x, c_y, c_x):
    
    # square_distance = math.pow((c_y - p_y),2) + math.pow((c_x - p_x),2)
    square_distance = np.square(c_y - p_y) + np.square(c_x - p_x)
    
    return square_distance

def belonging_ordered_keypoints_heatmap_flattened_indices(ref_y, ref_x, keypoints_heatmap_flattened_indices):
    keypoints_heatmap_indices_y, keypoints_heatmap_indices_x = get_row_col_indices_from_flattened_indices(keypoints_heatmap_flattened_indices)
    square_distances = cal_square_distance(ref_y, ref_x,keypoints_heatmap_indices_y, keypoints_heatmap_indices_x)
    smallest_dis_index = heapq.nsmallest(1, range(len(square_distances)), square_distances.__getitem__)
    true_keypoints_heatmap_flattened_index = keypoints_heatmap_flattened_indices[smallest_dis_index[0]]
    return true_keypoints_heatmap_flattened_index

def previous_index_selection_obc(index, topk_index):
    
    # return previous_y, previous_x
    pass


# def get_boxes_coordinates(object_center_y_indices, object_center_x_indices, boxes_wh, boxes_offsets, topk=1):
def get_boxes_coordinates(object_center_y_indices, object_center_x_indices, boxes_wh, boxes_offsets, topk=1):
    H,W,C = boxes_wh.shape    
    for index in range (C//2):
        if QUANTIZED_OUTPUT:
            # boxes_wh_h_list = ((boxes_wh[object_center_y_indices][object_center_x_indices][index*2] - BHWZeroPointValue) * BHWQuantizedScale)
            # boxes_wh_w_list = ((boxes_wh[object_center_y_indices][object_center_x_indices][index*2+1] - BHWZeroPointValue) * BHWQuantizedScale)
            boxes_wh_h_list = ((boxes_wh[object_center_y_indices, object_center_x_indices, index*2] - BHWZeroPointValue) * BHWQuantizedScale)
            boxes_wh_w_list = ((boxes_wh[object_center_y_indices, object_center_x_indices, index*2+1] - BHWZeroPointValue) * BHWQuantizedScale)
            
            # boxes_offsets_y_list = ((boxes_offsets[object_center_y_indices][object_center_x_indices][index*2] - BOFZeroPointValue) * BOFQuantizedScale)
            # boxes_offsets_x_list = ((boxes_offsets[object_center_y_indices][object_center_x_indices][index*2+1] - BOFZeroPointValue) * BOFQuantizedScale)
            boxes_offsets_y_list = ((boxes_offsets[object_center_y_indices, object_center_x_indices, index*2] - BOFZeroPointValue) * BOFQuantizedScale)
            boxes_offsets_x_list = ((boxes_offsets[object_center_y_indices, object_center_x_indices, index*2+1] - BOFZeroPointValue) * BOFQuantizedScale)
        else:
            boxes_wh_h_list = boxes_wh[object_center_y_indices, object_center_x_indices, index*2]
            boxes_wh_w_list = boxes_wh[object_center_y_indices, object_center_x_indices, index*2+1]
            
            boxes_offsets_y_list = boxes_offsets[object_center_y_indices, object_center_x_indices, index*2]
            boxes_offsets_x_list = boxes_offsets[object_center_y_indices, object_center_x_indices, index*2+1]
            
    ymin = object_center_y_indices + boxes_offsets_y_list - boxes_wh_h_list / 2.0
    xmin = object_center_x_indices + boxes_offsets_x_list - boxes_wh_w_list / 2.0
    ymax = object_center_y_indices + boxes_offsets_y_list + boxes_wh_h_list / 2.0
    xmax = object_center_x_indices + boxes_offsets_x_list + boxes_wh_w_list / 2.0
    
    ymin = np.clip(ymin, 0., H*1.0)
    xmin = np.clip(xmin, 0., W*1.0)
    ymax = np.clip(ymax, 0., H*1.0)
    xmax = np.clip(xmax, 0., W*1.0)
    
    # boxes = [ymin, xmin, ymin, xmax, ymax, xmax, ymax, xmin]
    # boxes = np.reshape(boxes,[-1,2,topk])

    boxes = [ymin, xmin, ymax, xmax]
    # boxes = np.reshape(boxes,[4,topk])
    
    return boxes

def get_object_reID(object_center_scores, object_center_y, object_center_x, detection_embeddings, topk=1):
    
    def comp_cosine_distance(vector1, vector2):
        return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))

    def comp_euc_distance(vector1, vector2):
        return np.sqrt(np.sum(np.square(vector1-vector2)))
    
    def id_tensor_smooth(detection_embeddings, G=2):
        kernel = np.ones((G,G), dtype='float32')/(G*G)
        result = cv2.filter2D(detection_embeddings, -1, kernel=kernel)
        return result

    # smooth id tensor
    detection_embeddings = id_tensor_smooth(detection_embeddings)

    reIDs = []

    detection_embeddings_topk = []
    for i in range(topk):
        detection_embeddings_topk.append(detection_embeddings[object_center_y[i], object_center_x[i]])
    
    # normalize topk id tensor
    if ID_TENSOR_NORMALIZE:
        topk_matrix = np.array(detection_embeddings_topk)
        topk_matrix_nm = (topk_matrix - topk_matrix.mean(axis=0))/1.08
        detection_embeddings_topk = list(topk_matrix_nm)

    #use both threshold value and minimum distance
    for i in range(topk):
        detection_embedding = detection_embeddings_topk[i]
        
        id = None
        if object_center_scores[i] > centerpoints_threshold_value:
            if len(reIDDict.keys()) == 0:
                id = 1
                reIDDict.update({id:detection_embedding})
            else:
                min_distance = 99999999
                min_id = -1
                for j in range(len(reIDDict.keys()), 0, -1):
                    vector1 = detection_embedding
                    vector2 = reIDDict.get(j)
                    distance = comp_euc_distance(vector1, vector2)
                    if distance < min_distance:
                        min_distance = distance
                        min_id = j
                if (min_distance < object_reID_eucDistance_threshold_value):
                    id = min_id
                    if DYNAMIC_UPDATE_ID_DIC:
                        reIDDict[j]=0.005*detection_embedding + 0.995*reIDDict[j]
                else:
                    if len(reIDDict.keys()) < MAX_PEOPLE:
                        id = len(reIDDict.keys()) + 1
                        reIDDict.update({id:detection_embedding})
                    else:
                        id = min_id
        else:
            id = -1
        reIDs.append(id)

    return reIDs

def coordinates_draw_reorder(points,topk=1):
    reordered_points = []
    
    return reordered_points    

def coordinates_decode(object_center_heatmap=None, boxes_wh=None, boxes_offsets=None, detection_embeddings=None, topk=1):
    refined_keypoints_coordinates = []
    if object_center_heatmap is not None:
        object_center_y,object_center_x, object_center_scores = get_object_center_coordinates_scores(object_center_heatmap, topk=topk)
    else:
        object_center_y,object_center_x, object_center_scores = None, None, None
    if (boxes_wh is not None) and (boxes_offsets is not None):
        boxes = get_boxes_coordinates(object_center_y, object_center_x, boxes_wh, boxes_offsets, topk=topk)
    else:
        boxes = None
    if (detection_embeddings is not None):
        reID = get_object_reID(object_center_scores, object_center_y, object_center_x, detection_embeddings, topk)
    else:
        reID = None

    return object_center_y, object_center_x, object_center_scores, boxes, reID
    
if __name__ == "__main__":
    
    tflite_model_path = "/workspace/MultiObjTracking/tflites/size8_modified.tflite"
    
    if not HEATMAP_SIMPLIFIED_DEBUG:
        interpreter = tf.lite.Interpreter(tflite_model_path)
    else:
        interpreter = tf.lite.Interpreter(tflite_model_path,experimental_preserve_all_tensors=True)
    
    interpreter.allocate_tensors()
    
    tensor_details = interpreter.get_tensor_details()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width = input_details[0]['shape'][2]
    height = input_details[0]['shape'][1]
    
    H, W = 384, 384
    
    centernet_scale = 4
    #cap = cv2.VideoCapture(3)

    times = 0
    frame_frequency = VIDEO_FRAME_FREQUENCY
    camera = cv2.VideoCapture('/workspace/MultiObjTracking/tflite_test/video/video1.mp4')
    global_path_2_frames = '/workspace/MultiObjTracking/tflite_test/videoDR_test/'

    size = (506,478)
    videowrite = cv2.VideoWriter('/workspace/MultiObjTracking/tflite_test/video/video_DR_test.mp4',0x7634706d,15,size)#20是帧数，size是图片尺寸
    img_array=[]
    

    while True:
        times = times + 1
        res, image = camera.read()
        if not res:
            print('not res , not image')
            break
        if times % frame_frequency != 0:
            continue
        if times < 40:
            continue
        if times > 440:
            break
        print(times)

        image_pic_name = global_path_2_frames + 'frame_' + str(times) + '.jpg'

        img = image
        imH, imW = img.shape[0] , img.shape[1]
        frame_copy = img.copy()
        
        Hscale, Wscale = imH/H, imW/W
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (H, W), cv2.INTER_AREA)
        
        # cv2.imwrite(image_pic_name, img_rgb)
        # sys.exit(1)
        
        img_rgb_copy = img_rgb.copy()
        
        img_rgb = img_rgb.reshape([1, H, W, 3])
        
        floating_model = (input_details[0]['dtype'] == np.float32)
        
        input_mean = 127.5
        input_std = 127.5
      
        if floating_model:
            if NORMALIZATION_INPUT:
                input_data = (np.float32(img_rgb) - input_mean) / input_std
            else:
                input_data = np.float32(img_rgb)
        else:
            input_data = img_rgb

        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        if OBJECT_WB_HEATMAP_REID_OUTPUT:
            boxes_offsets = interpreter.get_tensor(output_details[3]['index'])[0]
            boxes_wh = interpreter.get_tensor(output_details[0]['index'])[0]
            object_center_heatmap = interpreter.get_tensor(output_details[2]['index'])[0]
            #object_center_heatmap = interpreter.get_tensor(tensor_details[HEATMAP_SIMPLIFIED_DEBUG_LAYER_INDEX]['index'])[0]
            detection_embeddings = interpreter.get_tensor(output_details[1]['index'])[0]
        
        # np.set_printoptions(threshold=np.inf)
        # print(boxes_offsets.shape)
        # print(boxes_wh.shape)
        # print(object_center_heatmap.shape)
        # print(detection_embeddings.shape)
        # sys.exit(1)

        image_name = ('q_' if QUANTIZED_OUTPUT else 'non_q_') + ('normalization_' if NORMALIZATION_INPUT else 'non_normalization_') + 'sample_image'
        
        if SAVE_HEATMAP:
            heatmap_save(image_name,object_center_heatmap,boxes_offsets,boxes_wh)
        
        object_center_y, object_center_x, object_center_scores, boxes, reID = coordinates_decode(object_center_heatmap, boxes_wh, boxes_offsets, detection_embeddings, topk=TOPK_SELECTION)
        
        point_size = 1
        centerpoints_color = (0, 0, 255) # R
        boxes_points_color = (0,255,255) #Yellow
        line_color = (0,255,255) #Yellow
        boxes_line_color = (0, 255, 0) # G
        point_thickness = 8 # 可以为 0/4/8
        line_thickness = 2
        font_color = (0,0,0) #Yellow

        # boxes.shape: (4,topk)
        # boxes = [ymin, xmin, ymax, xmax]

      
        #draw center_points / boxex / reid
        Hscale=Hscale*centernet_scale
        Wscale=Wscale*centernet_scale
        id_set = set()
        for i in range(len(object_center_scores)):
            point = [object_center_y[i], object_center_x[i]]
            if object_center_scores[i] > centerpoints_threshold_value:
                #bbox
                leftTopPoint = swap_scale_points((boxes[0][i], boxes[1][i]),Hscale,Wscale)
                rightDownPoint = swap_scale_points((boxes[2][i], boxes[3][i]),Hscale,Wscale)
                #太小的框不打印
                if (rightDownPoint[1]-leftTopPoint[1]) < 100:
                    continue
                #一个id只打印一个框
                if reID[i] in id_set:
                    continue
                else:
                    id_set.add(reID[i])
                cv2.rectangle(frame_copy, leftTopPoint, rightDownPoint, line_color, line_thickness)

                #center point
                point_strided = swap_scale_points(point,Hscale,Wscale)
                cv2.circle(frame_copy, point_strided, point_size, centerpoints_color, point_thickness)
                
                #reID
                label = '%d' % (int(reID[i]))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                xmin = int(boxes[1][i] * Wscale)
                ymin = int(boxes[0][i] * Hscale)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame_copy, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame_copy, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        cv2.imwrite(image_pic_name, frame_copy)
        img_array.append(frame_copy)
        

    camera.release()
    for i in range(len(img_array)):#把读取的图片文件写进去
        videowrite.write(img_array[i])
    videowrite.release()
    print('end!')

