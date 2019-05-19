"""
Created on 5/12/2019

Reads trials one-by-one
Performs analysis

My initial approach. In the following versions, I will do small modifications.

@author: sinan
"""

## Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import cv2

from pathlib import Path

from scipy.io import loadmat
from scipy import signal


##
def load_excel_file(filename, sheetname='Sheet1'):
    """
    Read excel file content into a data frame
    :param filename: File path
    :param sheetname: Sheet name
    :return: pandas data frame
    """
    xl = pd.ExcelFile(filename)
    xl  = xl.parse(sheetname)
    return xl

##
def load_mat_file(filename, offsets = (0, 0, 0)):
    """
    Load data file. Organize and return.
    :param filename: File path
    :param offsets: DC offsets of 3D force signals
    :return: data, time vector, and sample rate.
    Ex:
        force, time, fs = load_data('trial8.mat')
    """
    # Load the data
    f = loadmat(filename)
    print("Available fields: ", list(f.keys()))  # f is a dictionary. Let's look at the keys

    # Create variables from loaded dictionary
    force = f['data'][:,0:6]
    fs = f['mySampleRate'][0]
    fs = int(fs)

    # Transform matrix for force data
    TF = [[1.117, -0.096747, 1.7516, 0.03441, -0.88072, 0.042127, -0.89026],
          [0.3134, 0.0041349, 0.0045219, -0.055942, 1.5273, 0.037719, -1.5227],
          [0.135, 1.4494, -0.061075, 1.6259, 0.083867, 1.5999, 0.0058155]]
    TF = np.array(TF)

    # Read force data
    force = np.concatenate((np.ones((len(force), 1)), force), axis=1)
    force = force @ TF.transpose()

    # Make baseband zero
    force[:, 0] = force[:, 0] - offsets[0]
    force[:, 1] = force[:, 1] - offsets[1]
    force[:, 2] = force[:, 2] - offsets[2]

    # Choose force channel for analysis
    force = force[:, 1]
    force = -force  # Invert the sign (increased as applied force increased)

    # Corresponding time vectors
    time = f['timestamps'][:]
    return force, time, fs

## Initial setup
maindir = Path.home() / 'Documents' / 'DATA' / 'SCCI'
subject = 'SC-EMG-08'
folder = 'behavior'

force_threshold = 0.05

offsets_file_name = maindir / 'files2call' / 'baseline_offsets_SC08.xlsx'
offsets_all = load_excel_file(offsets_file_name)

## Read all trials
trials_file_name = maindir / 'files2call' / 'mixed_trials_all_SC08.xlsx'
trials_all = load_excel_file(trials_file_name, sheetname='behavior')

## Process
number_of_reaches = []
trial_range = (10, 15)
count = 1
for row in trials_all.itertuples():
    if count < trial_range[0] or count > trial_range[1]:
        count += 1
        continue
    count += 1

    # Unpack trial's info
    session = str(row[2]).zfill(6)
    trial = str(row[3])
    start_time = row[4]
    end_time = row[5]

    # Define path for files to be read
    file_name = maindir / subject / session / folder / Path('trial' + trial + '.mat')
    video_name = maindir / subject / session / folder / Path('trial' + trial + '.avi')

    print('Processing......', subject, '-', session, '- trial', trial)

    # Load data
    data, time_data, fs = load_mat_file(file_name, offsets_all.iloc[:,2].values)

    # Filter data
    fch = 20.0  # high cut-off
    nyq = fs / 2
    wn = fch / nyq
    order = 6
    sos = signal.butter(order, wn, 'low', output='sos')
    fdata = signal.sosfiltfilt(sos, data, axis=0)

    # Find spikes
    rangeHeight = 0.1  # min peak height requirement
    minDistance = 0.2 * fs  # min distance between two peaks
    rangeWidth = 0.05 * fs  # min and max peak width requirement
    minProminence = 0.5  # min prominence
    threshold = None  # vertical distance between peak and its neighboring samples
    widthHeight = 0.5

    x = fdata
    peaks, properties = signal.find_peaks(x,
                                          height=rangeHeight,
                                          distance=minDistance,
                                          width=rangeWidth,
                                          rel_height=widthHeight,
                                          prominence=minProminence)

    # Examine the peak parameters on the signal
    contour_heights = x[peaks] - properties['prominences']
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.vlines(x=peaks, ymin=contour_heights, ymax=x[peaks])
    keys = ['width_heights', 'left_ips', 'right_ips']
    plt.hlines(*[properties.get(key) for key in keys], color="C2")
    plt.pause(0.5)
    plt.clf()

    number_of_reaches.append(peaks.shape[0])

    # VIDEO PROCESSING

    # Create video object
    cap = cv2.VideoCapture(str(video_name))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret is True:
            # Create our shapening kernel, it must equal to one eventually
            kernel_sharpening = np.array([[-1, -1, -1],
                                          [-1, 9, -1],
                                          [-1, -1, -1]])
            # Applying the sharpening kernel to the input image
            frameF = cv2.filter2D(frame, -1, kernel_sharpening)

            # Convert from colored to grayscale
            gray = cv2.cvtColor(frameF, cv2.COLOR_BGR2GRAY)

            # Convert from grayscale to black-and-white
            _, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

            # Define a region of interest for the white plastic tray (landmark)
            roi = np.zeros((600,800), np.uint8)
            roi[360:540, 320:560] = np.ones((180,240), np.uint8) * 255

            # Bitwise-AND the mask and the original image
            res = cv2.bitwise_and(thresh, thresh, mask=roi)

            # Contours (to find the big white blobs in the image)
            contours, hier = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if 5000 < cv2.contourArea(cnt) < 20000:
                    #print(cv2.contourArea(cnt))
                    cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 2)

            # This is to pick the desired white area, which is the plastic tray
            select_cnt = [cnt for cnt in contours if 5000 < cv2.contourArea(cnt) < 20000]
            if len(select_cnt) is not 1:
                print('Problem! Too many/few white areas detected')
                select_cnt = np.array([[300,300],[300,300]])
            else:
                xdim, ydim, zdim = select_cnt[0].shape
                select_cnt = select_cnt[0].reshape((xdim,zdim))

            # Coordinates of the landmark (upper left corner)
            min_x = np.min(select_cnt[:,0])
            min_y = np.min(select_cnt[:,1])

            # Coordinates of the desired window in which we will observe the movement of rat's paw
            x1, y1 = min_x, min_y - 200
            x2, y2 = min_x + 100, min_y

            # Add a rectangle
            img = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # img = cv2.line(img, (383,409), (511,409), (0,0,255), 5)

            # Display the final image
            if ret is True:
                cv2.imshow('frame', img)
                k = cv2.waitKey(25) & 0xFF
                if k == 27:
                    break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

plt.close()

# TODO: The camera angle changes from session to session. Therefore, the ROI has to be adjusted
#       accordingly. Adjust the size and location of the RED window. Look at other sessions
#       too see if there is a better one.