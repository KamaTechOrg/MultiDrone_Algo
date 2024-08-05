import importlib
import os
# Import the module dynamically
detectron2_evaluation_module = importlib.import_module("detectron2-gpu-utilization-evaluation")

# Access the function from the module
detectron2_evaluation = getattr(detectron2_evaluation_module, "detectron2_evaluation")


PATH_TO_DATA="data_for_evaluation"

if __name__=='__main__':
    # print the accuracy
    # for example: print(groundingDINO_evaluation(path_to_data))
    print(detectron2_evaluation(PATH_TO_DATA))
    #(0.9608865710560626, 0.5566465256797583)