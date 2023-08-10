# this video frame splitter uses the previously made text document to access files

import cv2
import io
import os
# import AAframe_fetcher as frmf
import json
from loguru import logger

logger.add("logging_videoToFrameSplitter_{time}.log")
mainToVidDict = {}
#startJsonFile = open("static\\datafile.json", 'w')
#json.dump(dict, startJsonFile)


path = ':home/zeus_videos'

location_to_save_frames = 'home\\v4\\zeus\\videops\\images'
if not os.path.isdir(location_to_save_frames):
    os.mkdir(location_to_save_frames)

count = 0  # for actual loop
logcount = 0 # for logging purposes

# making a list of each file path in the text document
with open("home\\v4\\zeus\\videops\\static\\vids_to_process.txt", 'r', encoding='utf-8') as f:
    file_paths = []
    for file_path in f:
        # print(file_path)
        file_paths.append(file_path.strip())  # using the .strip() to remove any whitespace

# traversing through the list
for file_path in file_paths:
    logger.info("Total number of files split: " + str(logcount))
    logcount += 1
# VIDEOS
    if file_path.endswith(('.mov', '.MOV', '.mpg', '.mp4', '.MP4', '.mkv',
                           '.mpeg4', '.wlmp', '.asf')):  # '.avi', 'AVI','.264', '.264+','.ts',
        #print('file_path', file_path, count)

        vidToImgDict = {}

        try:
            capture = cv2.VideoCapture(file_path)  # try to open video file
            # print(capture.isOpened())
        except:  # incase video file is corrupt/cannot be opened
            with io.open('home\\v4\\zeus\\videops\\static\\vids_not_to_process.txt', 'w', encoding='utf-8') as fn:  # opening in write mode
                fn.write(file_path + '\n')  # writing each file path on txt doc, each file path on a new line
            logger.error("Could not capture video " + file_path)
        # print('starting read')
        capture = cv2.VideoCapture(file_path)
        logger.info("Opening video file and starting to split into frames " + file_path)
        while(True):
            ret, frame = capture.read()  # capture frame by frame; if frame is read correctly, ret is True
            # The first value of the tuple is a Boolean indicating if we could read the
            # frame or not and the second value is the actual frame.

            vid_name = os.path.split(file_path)[-1]
            complete_name = os.path.join(location_to_save_frames + '\\' + vid_name + 'frame' + str(count) + ' .jpeg')
            # print('complete_name', complete_name)

            if ret:
                if (count % 100) == 0:  # selecting 1 in every 100 frames
                    vidToImgDict[complete_name] = []  # add to dictionary, will later
                    # print(vidToImgDict)


                    cv2.imwrite(complete_name, frame)  # saving frame in the previously created jpg file

                    count += 1  # increase count of no. of frames; stops duplicate frames/images
                else:
                    count += 1  # increase count of no. of frames; stops duplicate frames/images
                    continue
            else:
                break
        logger.info("Finished splitting video file into frames " + file_path)
        mainToVidDict[file_path] = vidToImgDict  # add to main dict
        # print(mainToVidDict)
        list = []
        list.append(mainToVidDict)
        jsonString = json.dumps(list, indent=2)
        with open('home\\v4\\zeus\\videops\\static\\allRawData.json', 'w+') as file:  # use w+ to overwrite anythign in the file
            file.write(jsonString)
            logger.info("Adding " + file_path + " to a list of video paths. Writing list of video paths to "
                                                "home\\v4\\zeus\\videops\\static\\allRawData.json.")
        capture.release()

    else:
        continue

for file_path in file_paths:
    logger.info("Total number of video/image files split: " + str(logcount))
    logcount += 1
# IMAGES
    # these are images so add it straight to frame folder
    if file_path.endswith(('.bmp', '.BMP', '.JPG', '.jpg', '.png', '.jpeg', '.tif', '.pgm')):
        # print('file_path', file_path, count)

        vidToImgDict = {}

        # https://www.adamsmith.haus/python/answers/how-to-write-a-file-to-a-specific-directory-in-python
        try:
            logger.info("Reading image " + file_path)
            image = cv2.imread(file_path)
        except:
            logger.error("Could not read image " + file_path)
        # print(image)
        # sometimes get error: Premature end of JPEG file --> could not put this file into vids not to process list
        vid_name = os.path.split(file_path)[-1]
        complete_name = os.path.join(location_to_save_frames + r'\\' + vid_name + 'frame' + str(count) + ' .jpeg')
        # print('complete_name', complete_name)
        # creating some jpg file with this location

        vidToImgDict[complete_name] = []  # add to dictionary, will later
        output = cv2.imwrite(complete_name, image)  # writing the image to the location
        count += 1  # increase count of no. of frames; stops duplicate frames/images
        # print(output)

        mainToVidDict[file_path].append(vidToImgDict)  # add to main dict
        # print(mainToVidDict)

        list2 = []
        list2.append(mainToVidDict)
        jsonString = json.dumps(list2, indent=2)
        with open('home\\v4\\zeus\\videops\\static\\allRawData.json', 'w+', encoding='utf-8') as file:  # use w+ to overwrite anythign in the file
            file.write(jsonString)
            logger.info("Adding " + file_path + " and its related images to a list. "
                                                "Writing the list to home\\v4\\zeus\\videops\\static\\allRawData.json.")

    else:
        continue
        # break

# write the mainToVidDict to a JSON file - this will be later accessed to add results
#jsonString = json.dumps(mainToVidDict, indent=2)
#with open('static\\datafile.json', 'w+') as file:
#    file.write(jsonString)


