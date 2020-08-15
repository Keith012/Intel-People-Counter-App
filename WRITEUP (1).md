# People Counter App Project Write-Up
## Explaining Custom Layers

Let's first understand what we mean by Custom Layers. Custom layers are layers that are not included in the list of the known layers.
The Intel Open Vino model optimizer customizes the list of unknown layers as custom.
What are Layers? A layer is the abstract concept of a math function that is selected for a specific purpose. It is also one of the  sequential series of building blocks within the neural network.

Let's understand what happens from the beginning when models are loaded in the model optimizer.
When a model is loaded into the open vino system, the model optimizer searches the list of known layers for each layer contained in the input model topology before building the model's internal representation, optimizing the model, and producing the Intermediate Representation files.

What is the intermediate representation? This is the neural network used only by the Inference Engine in OpenVINO abstracting the different frameworks and describing topology, layer parameters and weights.

The inference engine loads the layers rom the input model IR files into the specified device plugin which will search for a list known of layer implementation for the device. Layers not found in the list of layers(custom layers) in the device will be considered as unsupported by the Inference engine. 

When implementing a custom layer for the pretrained model in the  openvino toolkit, you will need to add extensions to both the inference engine and the model optimizer.

Custom Layer extensions for the model optimizer

The Model Optimizer starts with a library of known extractors and operations for each supported model framework which must be extended to use each unknown custom layer.
The needed custom layer extensions are:
1)Custom layer extarctor - This is responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer.
2)Custom Layer Operation - Responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters.

Custom Layer extension for the inference engine

Each device plugin includes a library of optimized implementations to execute known layer operations which must be extended to execute a custom layer. The custom layer extension is implemented according to the target device:

1)Custom Layer CPU Extension - A compiled shared library needed by the CPU Plugin for executing the custom layer on the CPU.

2)Custom Layer GPU Extension - OpenCL source code (.cl) for the custom layer kernel that will be compiled to execute on the GPU along with a layer description file (.xml) needed by the GPU Plugin for the custom layer kernel.

This is all made possible by the Model extension tool generator which generates template source code files for each of the extensions needed by the Model Optimizer and the Inference Engine

The process behind the coversion of custom layer can be simply concluded in these steps
1)Generate the Extension Template Files Using the Model Extension Generator
2)Using Model Optimizer to Generate IR Files Containing the Custom Layer
3)Edit the CPU,GPU Extension Template Files
4)Execute the Model with the Custom Layer



## Comparing Model Performance

I used 3 models namely:
1)ssd_mobilenet_v2_coco
2)faster_rcnn_inception_v2_coco
3)ssd_inception_v2_coco

This is how they compare
SIZE
1)The size of the frozen inference graph of the ssd mobilenet V2 before conversion was 69.7MB the post conversion size was 67.5MB
2)The size of the frozen inference graph of the faster rcnn inception model before conversion was 54.6MB the post-conversion size was 50.6MB
3)The size of the  frozen inference graph of the ssd inception model before conversion was 97.3MB the post-converison size was 95.2MB

INFERENCE TIME
The threshold was 0.5
Under the above threshold;
1)ssd mobilenet inference time was at an average of about 60ms pre-conversion and 70 post-conversion
2)faster rcnn inception model inference time was at at an average of 880ms pre-conversion and 930 post-conversion
3)ssd inception model was at an average inference time of 97ms pre-conversion and 94 post-conversion

ACCURACY
The accuracy in all models was better post-conversion however comparing all three models what I noticed is that;
1)The ssd inception model's accuracy wasn't good as it didn't detect most of the people in the frame, besides having a higher inference time then the ssd mobilenet

2)The faster rcnn inception model accuracy was better than the ssd inception and the ssd mobilenet models

3)The ssd mobilenet accuracy was better compared to the ssd inception but not better than the faster rcnn.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are in for example helping in implementing high level security in restricted area where the systsem will alert the security head in a company if a person or people were detected entering the area
It can also be used in the transport sector to control and count the number of people who are standing in the ticketing queue to prevenet congestion in the area where an alarm will be triggered if 2 or more people are detected on the counter.

It can be used to help in roll call in companies to help them know the exact number of employees who checked in.


Each of these use cases would be useful because they will make improve the security of areas and in our homes reduce congestion in supermarket queues which reduces the probability of people getting sick from unkown airborne diseases

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

Poor lighting will lead to decrease in accuracy as in my project most models missed people dressed in black and this could be improved by improving the lighting.

Lower model accuracy will lead to the system missing some people in the fame and this could be problematic if the frame is running live, if used to implement security, this could be baa as it would miss the robber in the live video

If the camera focal length/image size is poor this could lead to the edge system missing the frame as the bounding box would not be accurately drawn around the person.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

Model 1: [ssd_mobilenet_v2_coco] 

Source: [http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz]

I converted the model to an Intermediate Representation withthe following arguments

python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channel

The model was good for the app, I liked it more.
I tried to make the model better by reducing the threshold in my code because it missed somepeople in the frame when the threshold was high

Model 2: [faster_rcnn_inception_v2_coco]
Source: [http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz]

I converted the model to an Intermediate Representation with the following arguments
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model/faster_rcnn_inception_v2_coco/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

The model gave me a hard time, it took me several days to make it work.
I tried to make the model better by adding some extensions and making changes on the code as well as reducing the threshold


Model 3:[SSD_inception_v2_coco]
Source:[http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz]

I converted the model to an Intermediate Representation with the following arguments
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

The model was my least favourite it incorrectly counted the people on the frame
I tried to adjust the threshold and make some adjustments in my code but it still didn't meetmy expectations.

Final Conclusion

The best model, which guarantees the best results is the person-detection-retail-0013 by Intel

The model is downloaded in the following procedure

1)Download the pre-requisite library sourcing the openvino installation by

pip install requests pyyaml -t /usr/local/lib/python3.5/dist-packages && clear && 
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5

2)Navigating to the directory containing the Model downloader by
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader

3)Downloading the model with the following command
sudo ./downloader.py --name person-detection-retail-0013 --precisions FP16 -o /home/workspace
