import cv2
import io
import os

from loguru import logger

logger.add("logging_framesToProcess_{time}.log")

def get_list_of_frames(location_to_save_frames):
    frames_list = []  # initialising list of file paths for frames to be analysed
    count = 0  # logging purposes
    for root, dir_names, frame_names in os.walk(location_to_save_frames):  # traversing through all the files
        # for frm in frame_names[0::num]:
        for frm in frame_names:
            frames_list.append(os.path.join(root, frm))
    with io.open('home\\v4\\zeus\\videops\\static\\frames_to_process.txt', 'w', encoding='utf-8') as f:    # opening in write mode, using encoding to
        # account for unicode/non-ascii characters
        for each_frame in frames_list:
            count += 1
            f.write(each_frame + '\n')  # writing each file path on text document, each file path on a new line
            logger.info(each_frame + " has been indexed.")
            logger.info("Total number of frames indexed: " + str(count))


location_to_save_frames = 'home\\v4\\zeus\\videops\\images'
# num = 100  # for 1 in every num frames; numes
get_list_of_frames(location_to_save_frames)
