# Import the OS module
import os
import glob
import io  # have some non-ascii file names so need unicode when opening files
from loguru import logger

#logger.basicConfig(level=logger.DEBUG)
logger.add("logging_filePathIndexer_{time}.log")
# logger.add(sys.stderr, format="{time} {level} {} {message}", filter="my_module", level="INFO")



def get_list_of_files(path):  # For the given path, create a list of all files in the directory tree

    file_paths = []  # initialising list of file paths for files to be analysed
    unanalysed_file_paths = []  # initialising list of file paths for files that will not be analysed
    fileCount = 0  # for logger purposes
    for root, dir_names, file_names in os.walk(path):  # traversing through all the files
        for file in file_names:
            fileCount += 1
            if file.endswith(('.mov', '.MOV', '.mpg', '.mp4', '.MP4', '.mkv',
                              '.mpeg4', '.wlmp', '.asf',
                              '.bmp', '.BMP', '.JPG', '.jpg', '.png', '.jpeg', '.pgm')):
                # '.264', '.264+', '.avi', 'AVI', '.tif', '.ts',
                # only allowing image and video files to be listed
                file_paths.append(os.path.join(root, file))
                logger.info("Total no. of files processed: " + str(fileCount))
            else:
                unanalysed_file_paths.append(os.path.join(root, file))
                logger.info("Total no. of files processed: " + str(fileCount))

    with io.open('home\\v4\\zeus\\videops\\static\\vids_to_process.txt', 'w', encoding='utf-8') as f:    # opening in write mode, using encoding to
                                                                        # account for unicode/non-ascii characters
        for each_path in file_paths:
            try:
                f.write(each_path + '\n')  # writing each file path on text document, with each file path on a new line
                logger.info(each_path + " has been indexed and will be analysed.")
            except:
                logger.error(each_path + " could NOT be indexed.")

    with io.open('home\\v4\\zeus\\videops\\static\\vids_not_to_process.txt', 'w', encoding='utf-8') as fn:  # opening in write mode
        for each_path in unanalysed_file_paths:
            try:
                fn.write(each_path + '\n')  # writing each file path on text document, with each file path on a new line
                logger.info(each_path + " has been indexed but will not be analysed")
            except:
                logger.error(each_path + " could NOT be indexed.")

    f.close()
    fn.close()

path = ':home/zeus_videos'
get_list_of_files(path)
