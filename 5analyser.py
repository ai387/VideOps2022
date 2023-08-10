"""
Factory Pattern:
when you are about to instantiate, let's encapsulate that instantiation, so that we can make it uniform across all
places
you can use the factory to instantiate, and the factory is responsible for instantiating properly
"""
#from audioop import mul
#from concurrent.futures import thread
import io
import os
import sys
from onnx_to_tensorrt import object_detector
from network import mainClassifierNetwork
import tensorrt as trt

from abc import ABCMeta, abstractmethod

# using one of the below to run several models concurrently 
#import asyncio
#import multiprocessing
#import threading

from loguru import logger


MODELS_TO_PROCESS = ["Object Detector", "Image Classifier"]


logger.add("logging_analyser_{time}.log")

sys.path.insert(1, os.path.join(sys.path[0], "../.."))

# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER = trt.Logger()

location_to_save_frames = r'C:\Users\arzina.ismaili\Pictures'

""" 
def make_config_file():
    logger.info("Making config file.")
    with io.open("config_file.txt", 'w', encoding='utf-8') as f:
        f.write(
                "Object Detector\nImage Classifier\n"
                )  # can add more models manually

# load the configuration file
def load_config_file(config_filename) -> list:  # list o/p of the models in the config file
    make_config_file()
    logger.info("Loading config file.")
    modelNames = []  # first making a list of models shown in the config file
    with io.open(config_filename, 'r', encoding='utf-8') as f:
        for model in f:
            modelNames.append(model.strip())  # use .strip when making the dictionary to remove any whitespace
    logger.info("Config file has been read.")
    # add list models=[...,...,...] to a dictionary
    # return {"models": modelNames}  # {"models": ["objectDetector, wavebandDetector,..."]}
    return modelNames

 """



class IModelClass(metaclass=ABCMeta):  # abstract class --> cannot create any instances of it
    @abstractmethod
    def __init__(self):
        # self.name = name  # self.name is an attribute and is equal to the 'name' that is passed into the function
        # self.image = image
        """Interface method"""

    @abstractmethod
    def process_frame_method(self):
        """Interface Method to process frame through neural net
        Interface Method - it is a definition of the signature (it has this function/method, but dont have to say
        what this method does); it mentions that each instance of a class/model has to implement this method
        """

class ObjectDetector(IModelClass):  # inherits/implements the IModelClass

    def __init__(self):
        """text"""

    def process_frame_method(self):

        MODELS_TO_PROCESS.remove('Object Detector')
        if len(MODELS_TO_PROCESS) == 0:
            logger.info("No more models to run. All processing complete.") 
        else:
            factory = read_factory(MODELS_TO_PROCESS)
            main(factory)
            
        # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
        logger.info('Running Object Detector --> running code in python file: onnx_to_tensorrt.py')
        frame_paths = frmf.frame_fetcher("home\\v4\\zeus\\videops\\static\\frames_to_process.txt")
        os.system('onnx_to_tensorrt.py')
        object_detector(frame_paths)

class ImageClassifier(IModelClass):  # inherits/implements the IModelClass

    def __init__(self):
        """text"""

    def process_frame_method(self):

        MODELS_TO_PROCESS.remove('Image Classifier')
        if len(MODELS_TO_PROCESS) == 0:
            logger.info("No more models to run. All processing complete.") 
        else:
            factory = read_factory(MODELS_TO_PROCESS)
            main(factory)
            
        logger.info('Running Image Classifier --> running code in python file: network.py')
        os.system('network.py')
        frame_paths = frmf.frame_fetcher("home\\v4\\zeus\\videops\\static\\frames_to_process.txt")
        mainClassifierNetwork(frame_paths, 'resnet50_final.trt')



class ModelFactory(metaclass=ABCMeta):
    """Factory that will create models according to the config file"""

    @abstractmethod
    def get_model(self) -> IModelClass:
        """interface method to get model e.g.ModelTester"""

class objectDetectorFactory(ModelFactory):
    def get_model(self) -> IModelClass:
        return ObjectDetector()

class imageClassifierFactory(ModelFactory):
    def get_model(self) -> IModelClass:
        return ImageClassifier()





def read_factory(modelNames) -> ModelFactory:
    """create models according to the config file"""
    logger.info("Reading Factory")
    factories = {
        "Object Detector": objectDetectorFactory(),
        "Image Classifier": imageClassifierFactory(),
        }

    if len(modelNames) == 0:
        logger.info("No more models to run. All processing complete.") 
    else:
        pass
    
    for modelName in modelNames:
        #logger.info(modelName)

        if modelName in factories.keys():
            logger.info("This modelName "+modelName+" is in the dictionary of available models.")
            return factories[modelName]

        else:
            logger.critical("Unknown model, please add to code before continuing.")
    # logger.info("Models to run include: " + str(i) for i in modelNames)
    
    """ try:
        #loop = asyncio.get_event_loop()
        logger.info("Attempting to run all models.")
        #loop.run_until_complete(asyncio.gather(objectDetectorFactory(), imageClassifierFactory()))
        #Thread1 = threading.Thread(target=objectDetectorFactory())
        #Thread2 = threading.Thread(target=imageClassifierFactory())
        #Thread1.start()
        #Thread2.start()
        #process1 = multiprocessing.Process(target=objectDetectorFactory())
        #process1.start()
        #factories["Image Classifier"]
        #factories["Object Detector"]
        print(modelNames)
        for i in modelNames:
            if i in factories.keys():
                print(i)
                #out = modelNames.pop(i)
                return factories[i]
            else:
                logger.critical("Unknown model, please add to code before continuing.")
        
    except:
        logger.critical("Model(s) did not run. Check code for errors and ensure model is in code.") """
    
    
"""     for modelName in modelNames:
        #logger.info(modelName)
        if modelName in factories.keys():
            logger.info("This modelName "+modelName+" is in the dictionary of available models.")
            
            THREAD = threading.Thread(target=out)
            THREAD.start()

        else:
            logger.critical("Unknown model, please add to code before continuing.") """

    

def main(fac: ModelFactory) -> None:

    logger.info("Retrieving the model.")
    model = fac.get_model()

    logger.info("Will process frames now.")
    model.process_frame_method()


if __name__ == '__main__':
    logger.info("Starting main() code.")
    # modelNames = load_config_file("config_file.txt")
    # modelNames = ['Object Detector', 'Image Classifier']

    logger.info("Starting read_factory() code.")
    factory = read_factory(MODELS_TO_PROCESS)

    main(factory)
