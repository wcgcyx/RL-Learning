import cv2


def render(environment):
    frame = grab_frame(environment)
    cv2.imshow("Display", frame)
    cv2.waitKey(1)


def grab_frame(environment):
    # Get RGB rendering of env
    rgbArr = environment.physics.render(480, 600, camera_id=0)
    # Convert to BGR for use with OpenCV
    return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)