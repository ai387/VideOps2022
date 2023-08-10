#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function
import json
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
import io

from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
from downloader import getFilePath
import AAframe_fetcher as frmf

from loguru import logger

logger.add("logging_onnxToTensorrt_ObjectDetector_{time}.log")

TRT_LOGGER = trt.Logger()




def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    #print(bboxes, confidences, categories)
    try:
        for box, score, category in zip(bboxes, confidences, categories):
            x_coord, y_coord, width, height = box
            left = max(0, np.floor(x_coord + 0.5).astype(int))
            top = max(0, np.floor(y_coord + 0.5).astype(int))
            right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
            bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

            draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
            draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)
    except TypeError:
        logger.error("COULD NOT PROCESS IMAGE because 'NoneType' object is not iterable")

    return image_raw

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 608, 608]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def makeObjDetDict(bboxes, confidences, categories):
    logger.info("Making dictionary of results for the frame.")
    #logger.info("Values for the boxes, confidences, categories are: " + bboxes, confidences, categories)
    try:
    #for box, score, category in zip(bboxes, confidences, categories):
    #if zip(bboxes, confidences, categories):
        #if box is None:  # or score or category
        objDetDictList = []

        for box, score, category in zip(bboxes, confidences, categories):
            x_coord, y_coord, width, height = box
            dict = {
                "x_coord": float(x_coord),
                "y_coord": float(y_coord),
                "width": float(width),
                "height": float(height),
                "score": float(score),
                "category": int(category)
                }
            objDetDictList.append(dict)

    except:  #TypeError
        logger.error("COULD NOT PROCESS IMAGE because 'NoneType' object is not iterable")
        objDetDictList = []


    return objDetDictList

#def main():
def object_detector(frame_paths):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""
    for path in frame_paths:

        # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
        onnx_file_path = 'yolov3.onnx'
        engine_file_path = "yolov3.trt"
        # Download a dog image and save it to the following file path:
        input_image_path = getFilePath(path)
        # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
        input_resolution_yolov3_HW = (608, 608)
        # Create a pre-processor object by specifying the required input resolution for YOLOv3
        preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
        # Load an image from the specified input path, and return it together with  a pre-processed version
        image_raw, image = preprocessor.process(input_image_path)
        # Store the shape of the original input image in WH format, we will need it for later
        shape_orig_WH = image_raw.size

        # Output shapes expected by the post-processor
        output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]
        # Do inference with TensorRT
        trt_outputs = []
        with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            # Do inference
            print('Running inference on image {}...'.format(input_image_path))
            logger.info('Running inference on image {}...'.format(input_image_path))
            # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            inputs[0].host = image
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

        postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                              "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  # A list of 9 two-dimensional tuples for the YOLO anchors
                                               (59, 119), (116, 90), (156, 198), (373, 326)],
                              "obj_threshold": 0.6,                                               # Threshold for object coverage, float value between 0 and 1
                              "nms_threshold": 0.5,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                              "yolo_input_resolution": input_resolution_yolov3_HW}

        postprocessor = PostprocessYOLO(**postprocessor_args)

        # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
        boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
        # Draw the bounding boxes onto the original input image and save it as a PNG file
        obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)

        #img_name = os.path.split(path)[-1]
        #output_image_path = os.path.join(location_to_save_frames + img_name)
        #print(output_image_path)
        output_image_path = input_image_path  # replace original image with analysed image

        obj_detected_img.save(output_image_path, 'PNG')
        print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))
        logger.info('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))
        #os.chdir(r'C:\Users\arzina.ismaili\OneDrive - Chess Dynamics Ltd\Desktop\PycharmProjects\VideoProcessor\frame_info_docs')

        objDetDict = makeObjDetDict(boxes, scores, classes)

        frmf.writeJsonModelDict('allRawData.json', objDetDict, path)
        logger.info("Written results from the object detector model as a dictionary to allRawData.json")

        if len(objDetDict) != 0:  # check if empty dictionary or not
            category = objDetDict[0]['category']
            frmf.writeJsonDetecttoFilenames("ObjDetData.json", str(category), path)  # adding to file
            logger.info("Written results from the object detector model as a dictionary to ObjDetData.json")
        else:
            logger.info("No results from the object detector model to add to ObjDetData.json")



#if __name__ == '__main__':
#    main()
#frame_paths = frmf.frame_fetcher("static\\frames_to_process.txt")

#object_detector(frame_paths)