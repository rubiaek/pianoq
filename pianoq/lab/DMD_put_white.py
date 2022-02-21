""" Adapted from dmd_binary_checkerboard_example.py and example_helper.py example scripts of ajile"""
import os 
import sys 
import numpy as np
try:
    import ajiledriver as aj
except ImportError:
    print("Can't load ajiledriver for DMD")

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0])
cur_dir = os.path.dirname(os.path.abspath(__file__))
WHITE_PATH = os.path.join(cur_dir, "../data/white.png")
BLA_PATH = os.path.join(cur_dir, "../data/bla.png")


def get_image(deviceType):
        
    image = aj.Image(1)
    # TODO: make this work from memory 
    # white_img = np.zeros(shape=(imageHeight, imageWidth))
    # srcNumChan=0; srcBitDepth=8; srcMajorOrder=aj.ROW_MAJOR_ORDER; dstDeviceType=deviceType
    # image.ReadFromMemory(white_img, imageHeight, imageWidth, srcNumChan, srcBitDepth, srcMajorOrder, dstDeviceType)
    image.ReadFromFile(WHITE_PATH, deviceType)
    return image


def get_project(sequenceID=1, sequenceRepeatCount=0, frameTime_ms=1e4, components=None):
    
    project_name = 'sample_white'
    project = aj.Project(project_name)

    # set the project components and the image size based on the DMD type
    project.SetComponents(components)
    dmdIndex = project.GetComponentIndexWithDeviceType(aj.DMD_4500_DEVICE_TYPE)
    if dmdIndex < 0: 
        dmdIndex = project.GetComponentIndexWithDeviceType(aj.DMD_3000_DEVICE_TYPE)
    imageWidth = components[dmdIndex].NumColumns()
    imageHeight = components[dmdIndex].NumRows()
    deviceType = components[dmdIndex].DeviceType().HardwareType()
    
    image = get_image(deviceType)
    project.AddImage(image)
    
    numImages = 1
    
    # create the sequence
    project.AddSequence(aj.Sequence(sequenceID, project_name, deviceType, aj.SEQ_TYPE_PRELOAD, sequenceRepeatCount))

    # create a single sequence item, which all the frames will be added to
    project.AddSequenceItem(aj.SequenceItem(sequenceID, 1))

    # create the frames and add them to the project, which adds them to the last sequence item
    for i in range(numImages):
        frame = aj.Frame()
        frame.SetSequenceID(sequenceID)
        frame.SetImageID(1)
        frame.SetFrameTimeMSec(frameTime_ms)
        project.AddFrame(frame)

    return project


def main():
    hs = aj.HostSystem()
    hs.SetConnectionSettingsStr('192.168.200.1', '255.255.255.0', '0.0.0.0', 5005)
    hs.SetCommunicationInterface(aj.USB3_INTERFACE_TYPE)
    # hs.SetUSB3DeviceNumber(deviceNumber)
    
    ret = hs.StartSystem()
    if ret != aj.ERROR_NONE:
        print(f"Error! {ret}")
    
    sequenceID = 1
    sequenceRepeatCount = 0
    frameTime_ms = 1e3  # 1 sec - compromise between fast script ending, and not putting a million images on DMD
    project = get_project(sequenceID, sequenceRepeatCount, frameTime_ms=frameTime_ms, components=hs.GetProject().Components())
    
    sequence, wasFound = project.FindSequence(sequenceID)
    if not wasFound:
        print("ERROR! sequence not found")
        sys.exit(-1)
        
    componentIndex = hs.GetProject().GetComponentIndexWithDeviceType(sequence.HardwareType())

    # stop any existing project from running on the device
    hs.GetDriver().StopSequence(componentIndex)
    
    hs.GetDriver().LoadProject(project)
    
    timeout_ms=5000
    ret = hs.GetDriver().WaitForLoadComplete(timeout_ms)
    if not ret:
        print("Timeout loading project!")
        sys.exit(-1)

    for sequenceID, sequence in project.Sequences().iteritems():
        # if using region-of-interest, switch to 'lite mode' to disable lighting/triggers and allow DMD to run faster
        roiWidthColumns = sequence.SequenceItems()[0].Frames()[0].RoiWidthColumns()
        if roiWidthColumns > 0 and roiWidthColumns < aj.DMD_3000_IMAGE_WIDTH_MAX:
            hs.GetDriver().SetLiteMode(True, componentIndex)
            
        # run the project
        if frameTime_ms > 0:
            print ("Starting sequence %d with frame rate %f and repeat count %d" % (sequence.ID(), frameTime_ms, sequenceRepeatCount))

        hs.GetDriver().StartSequence(sequence.ID(), componentIndex)

        # wait for the sequence to start
        print ("Waiting for sequence %d to start" % (sequence.ID(),))
        while hs.GetDeviceState(componentIndex).RunState() != aj.RUN_STATE_RUNNING: 
            pass

        if sequenceRepeatCount == 0:
            try:
                input("Sequence repeating forever. Press Enter to stop the sequence")
                hs.GetDriver().StopSequence(componentIndex)
            except Exception as e:
                print("ERROR!!")
                print(e)
                hs.GetDriver().StopSequence(componentIndex)

        print("Waiting for the sequence to stop.")
        try:
            while hs.GetDeviceState(componentIndex).RunState() == aj.RUN_STATE_RUNNING:
                pass
        except Exception as e:
            print("ERROR!!")
            print(e)
            hs.GetDriver().StopSequence(componentIndex)


if __name__ == "__main__":
    main()
