"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def draw_boxes(frame, result, prob_threshold, width, height): # Draws bounding boxes on the frame
    person_in_frame = 0
    for box in result[0][0]:
        conf = box[2]
        if conf >= prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame,(xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            person_in_frame += 1
            
    return frame, person_in_frame
    
def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Input arguments
    model = args.model
    prob_thresh = args.prob_threshold
    cpu_extension = args.cpu_extension
    device = args.device
    filio = args.input
    
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = prob_thresh
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, cpu_extension, "CPU")
    net_input_shape = infer_network.get_input_shape()
    #print(net_input_shape)   
    
    ### TODO: Handle the input stream ###
    #Flag for single images
    input_image = False
    
    #Checks for live feed
    if filio == "CAM":
        filio = 0
    
    #Checks for input image
    elif filio.endswith('.jpg') or filio.endswith('.bmp'):
        input_image = True
        
    #Checks for video file
    else:
        isFile = os.path.isfile("filio")
        #print(isFile)
        
    cap = cv2.VideoCapture(filio)
    cap.open(filio)
    
    #Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    #Initialize the variables
    current_count = 0  
    total_count = 0
    last_count = 0
    start_time = 0 
    duration  = 0
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)    

        ### TODO: Pre-process the image as needed ###
        image_p = cv2.resize(frame,(net_input_shape[3],net_input_shape[2]))
        image_p = image_p.transpose((2,0,1))
        image_p = image_p.reshape(1, *image_p.shape)
        
        #print(image_p)
        
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(image_p)
        inf_beg = time.time()

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            dur = time.time() - inf_beg

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            
            ### TODO: Extract any desired stats from the results ###
            frame_box, person_in_frame = draw_boxes(frame, result, prob_threshold, width, height)
            
            
            # Display inference time
            message = "Macharia_Mwangi, inference time: {:.3f}ms".format(dur * 1000) 
            cv2.putText(frame_box, message, (15,15), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (0,255,0), 1)
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            # When a person enters the video
            current_count = person_in_frame
            if current_count > last_count:
                start_time = time.time()
                total_count += current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))
                
            stat = "The current count is: {0:d}".format(current_count)
            cv2.putText(frame_box, stat, (15,30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (255,0,0), 1)
            
            stat2 = "The total count is: {0:d}".format(total_count)
            cv2.putText(frame_box, stat2, (15,45), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (0,0,255), 1)
                
            # Person duration in the video calculation
            if current_count < last_count:
                duration = int(time.time() - start_time)
                client.publish("person/duration", json.dumps({"duration": duration}))
                #print('duration_terminal',duration)
                
            Total_duration = "The duration is: {0:d}".format(duration)
            cv2.putText(frame_box, Total_duration,(15,60), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (255,0,0), 1)
            
            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count
            
            if key_pressed == 27:
                break
                     
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        

        ### TODO: Write an output image if `single_image_mode` ###
        if input_image:
            cv2.imwrite("output_image.jpg", frame)


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
