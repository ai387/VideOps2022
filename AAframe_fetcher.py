import json
import os
from loguru import logger

logger.add("logging_AAframeFetcher_{time}.log")
# fetches frames form the list of frames

def frame_fetcher(name_text_doc):
    logger.info("Fetching frames.")
    with open(name_text_doc, 'r', encoding='utf-8') as f:
        frame_paths = []
        for frame_path in f:
            frame_paths.append(frame_path.strip())  # using the .strip() to remove any whitespace
    logger.info("Returning list of frames.")
    return frame_paths


# when you call this function, you can do list = frame_fetcher(name_text_doc):
# frames_list = frame_fetcher("static\\frames_to_process.txt")
# print(frames_list[8])


def writeJsonModelDict(file, modelDict, img_path):
    # First load existing data into a dict.
    with open(file, 'r+', encoding='utf-8') as f:
        logger.info(r"Opened static\allRawData.json.")
        file_data = json.load(f)
        # print(file_data)

    vid_name = (os.path.split(img_path)[-1]).split('frame')[0]
    # key = ('W:' + vid_name)
    for eachkey in file_data[0].keys():
        if eachkey.endswith(vid_name):
            key = eachkey
    #print(key)
    #print(img_path)
    #print(file_data[0][key])
    #print(file_data[0][key][img_path])

    try:
        if key.endswith(vid_name):
            file_data[0][key][img_path].append(modelDict)
        # file_data[key].append(modelDict)
        logger.info("Appended a dictionary with data regarding " + img_path + " to a list")
    except:
        logger.error("Not appended any data regarding " + img_path + " to a list")

    # convert back to json.
    # json.dump(file_data, file, indent=2)
    # f.seek(0)  # rewind
    with open(file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(file_data, indent=2))
        logger.info("Writing new data to json file " + file)
        # print(file_data)
        # f.truncate()  # deal with the case where the new data is smaller than the previous
        # print("dumped dict to json file")
        f.close()


def writeJsonClasstoFilenames(file, prediciton, img_path):
    # First load existing data into a dict.
    with open(file, 'r+', encoding='utf-8') as f:
        logger.info("Opened static\classificationDataFile.json")
        file_data = json.load(f)
        # print(file_data)

    vid_name = (os.path.split(img_path)[-1]).split('frame')[0]
    #vid_path = ('W:' + vid_name)
    #print(vid_path)
    # tuple = (vid_path, img_path)
    #logger.info("Made tuple containing video path: ", vid_path)
    # vid_name = (os.path.split(img_path)[-1]).split('_')[-1]
    # vid_path = ('W:' + vid_name.split(' .jpeg')[0])

    # print(key)
    # print(img_path)

    try:
        if vid_name not in file_data[0][prediciton]:
            file_data[0][prediciton].append(vid_name)
            logger.info("Appended classification data regarding " + img_path + " to a list.")
        else:
            logger.info("Classifier data regarding video " + vid_name + " already exists in this dictionary.")
            pass
    except:
        logger.error("Not appended any data regarding " + img_path + " to a list.")

    # convert back to json.
    # json.dump(file_data, file, indent=2)
    # f.seek(0)  # rewind
    with open(file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(file_data, indent=2))
        logger.info("Writing new classification data to json file ", file)
        # print(file_data)
        # f.truncate()  # deal with the case where the new data is smaller than the previous
        # print("dumped dict to json file")
        f.close()


def writeJsonDetecttoFilenames(file, category, img_path):
    # First load existing data into a dict.
    with open(file, 'r+', encoding='utf-8') as f:
        logger.info("Opened static\ObjDetData.json")
        file_data = json.load(f)

    vid_name = (os.path.split(img_path)[-1]).split('frame')[0]

    try:
        if vid_name not in file_data[0][category]:
            file_data[0][category].append(vid_name)
            logger.info("Appended detector data regarding " + img_path + " to a list.")
        else:
            logger.info("Detector data regarding video " + vid_name + " already exists in this dictionary.")
            pass
    except:
        logger.error("Not appended any data regarding " + img_path + " to a list.")


    with open(file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(file_data, indent=2))
        logger.info("Writing new detector data to json file ", file)

        f.close()

# writeJsonClasstoFilenames("static\\classificationDataFile.json", "1", r'C:\Users\arzina.ismaili\Pictures\frame0_V4 Final .mov .jpeg')
