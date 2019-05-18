import numpy as np
import h5py
import cv2
import openpyxl
import pickle


def load_h5py_file(fname, offsets = [0, 0, 0]):
    """
    Load data file. Organize and return data, time vector, and sample rate.
    Ex:
        neural, emg, force, time, fs = load_data('trial8.mat')
    """
    # Load the data
    f = h5py.File(fname, 'r')  # r for read only
    print("Available fields: ", list(f.keys()))  # f is a dictionary. Let's look at the keys

    # Create variables from loaded dictionary
    neural_data = f['ripple_data'][:,0:32]
    emg_data = f['ripple_data'][:,32:]
    force_data = f['data'][0:6,:].transpose()
    fs = f['mySampleRate'][:]

    # Transform matrix for force data
    TF = [[1.117	, -0.096747,	 1.7516, 0.03441, -0.88072, 0.042127, -0.89026],
          [0.3134, 0.0041349, 0.0045219, -0.055942, 1.5273, 0.037719,-1.5227],
          [0.135	, 1.4494, -0.061075, 1.6259, 0.083867, 1.5999, 0.0058155]]
    TF = np.array(TF)

    # Read force data
    force_data = np.concatenate((np.ones((len(force_data),1)), force_data), axis=1)
    force_data = force_data @ TF.transpose()

    # Make baseband zero
    force_data[:,0] = force_data[:,0] - offsets[0]
    force_data[:,1] = force_data[:,1] - offsets[1]
    force_data[:,2] = force_data[:,2] - offsets[2]

    # Use sent and received pulse signals to allign DAQ and RIPPLE data
    pulse_sent = f['data'][6,:].transpose()
    ps_ind, = np.nonzero(pulse_sent>1)
    ps_ind = ps_ind[0]

    pulse_received = f['ttl_data'][:,0]
    pr_ind, = np.nonzero(pulse_received>2000)
    pr_ind = pr_ind[0]

    p_diff = ps_ind - pr_ind

    # Align data
    if p_diff > 0:
        pulse_sent = np.concatenate((pulse_sent[p_diff:], np.zeros((p_diff,))), axis=0)
        trailing = np.mean(force_data[-int(fs*0.1):], axis=0) * np.ones((p_diff,1))
        force_data = np.concatenate((force_data[p_diff:,:], trailing))
    else:
        pulse_sent = np.concatenate((np.zeros((-p_diff,)), pulse_sent[:p_diff]), axis=0)
        leading = np.mean(force_data[:int(fs * 0.1)], axis=0) * np.ones((-p_diff, 1))
        force_data = np.concatenate((leading, force_data[:p_diff,:]))

    # Choose force channel for analysis
    force_data = force_data[:,1]
    force_data = -force_data # Invert the sign (increased as applied force increased)

    # Choose EMG data
    emg_data = emg_data[:,(5,15)]-emg_data[:,(23,25)]

    # Re-order EMG data so that 1. Dorsal 2. Biceps 3. Ventral 4. Triceps
    positions3 = (0,1)
    emg_data = emg_data[:,positions3]

    # Corresponding time vectors
    time = f['ripple_time'][:]
    return neural_data, emg_data, force_data, time, fs

##
def play_video_file(fname : str):
    """
    Plays video file and closes the window when finished.
    :type fname: string
    """
    cap = cv2.VideoCapture(fname)
    fps = cap.get(5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0, 0, 0)
    lineType = 2

    myvideo = []
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.putText(gray, 'Time: ' + str(round(cap.get(0) / 1000, 2)),
                        (10, 30),
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            cv2.putText(gray, 'Frame: ' + str(int(cap.get(1))),
                        (10, 70),
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            myvideo.append(gray)
            #cv2.imshow('frame', gray)
            #cv2.waitKey(10)
            #if cv2.waitKey(delay=2) & 0xFF == ord('q'):
            #    break
        else:
            break

    cap.release()

    if fps < 60:
        for frame in myvideo:
            cv2.imshow('frame', frame)
            cv2.waitKey(10)
    else:
        for ind, frame in enumerate(myvideo):
            if ind % 3 == 0:
                cv2.imshow('frame', frame)
                cv2.waitKey(10)
            else:
                continue
    cv2.destroyAllWindows()


##
def load_excel_file(fname, sheet_name='Sheet1', cell_range=None):
    """
    Reads data from from .xls and .xlsx file and returns data as numpy array
    :param fname: The excel file name. e.g.: 'my_excel_file.xlsx'
    :param sheet_name: Sheet name. e.g.: 'Sheet1'
    :param cell_range: Range to be read. e.g.: 'B2:F14'
    :return: A numpy array that is holding the values
    """
    #DEPRICATED CODE
    #d = cell_range.split(:))
    #row_size = int(d[1][1:]) - int(d[0][1:]) + 1
    #col_size = (ord(d[1][0].lower()) - 96) - (ord(d[0][0].lower()) - 96) + 1
    #cell_data = np.zeros([row_size, col_size])

    wb = openpyxl.load_workbook(fname)
    sheet = wb[sheet_name]

    big_list = []
    for row in sheet[cell_range]:
        small_list = []
        for cell in row:
            #print(cell.coordinate, cell.value)
            small_list.append(cell.value)
        big_list.append(small_list)

    return np.array(big_list)


## Using Python's pickle module to save and load objects
def save_obj(obj, name ):
    with open('Obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('Obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)