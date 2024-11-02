from pathlib import Path

import torch.cuda
from ultralytics import YOLO
from boxmot import BoTSORT

from src.inference import inference
from src.utils import (
    init_roi,
    is_person_in_roi,
    init_person,
    update_person_info,
    NpEncoder,
    checkIntersection,
    save_output,
    adjust_gamma,
    get_dets,
    parse_opt
)

def main():

    args = parse_opt()
    people_det = dict()

    # ---------- FILE PATHS ----------
    OUTPUT_FILENAME = args.output_json_path
    VIDEO_PATH = args.input_video_path
    CONFIGURATION_FILE_PATH = args.configuration_path

    # ---------- VIDEO INFO ----------
    FRAME_TO_SKIP = 2
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
    FRAME_RATE = int(video_info.fps / FRAME_TO_SKIP)
    INFERENCE_RATE = FRAME_RATE * 2  # one inference every 2 seconds
    EXIT_COUNT = FRAME_RATE  # value for considering a person out of the roi
    WIDTH_RESOLUTION = video_info.width
    HEIGHT_RESOLUTION = video_info.height
    print(f"Video information:\n"
          f"\t- width: {WIDTH_RESOLUTION}\n"
          f"\t- height: {HEIGHT_RESOLUTION}\n"
          f"\t- fps: {FRAME_RATE}\n")

    # ---------- ROI INFO AND CONFIGURATION ----------
    roi_list = init_roi(CONFIGURATION_FILE_PATH, video_info=video_info)
    roi1_passages = 0
    roi2_passages = 0
    roi1_count = 0
    roi2_count = 0

    # ---------- COLORS AND CONFIGURATION ----------
    # colors are in BGR format
    ROI_COLOR = (0, 0, 0)  # black
    ROI1_COLOR = (190, 115, 0)  # blue
    ROI2_COLOR = (50, 170, 120)  # green
    BBOX_COLOR = (0, 0, 255)  # red
    WHITE = (255, 255, 255)  # white

    # information for drawing part
    THICKNESS = 2
    ROI_THICKNESS = 3
    FONTSCALE = 0.6
    DX, DY = 5, 5  # delta x and delta y values used for drawing

    print(f"Regions of interest information:")
    for i, roi in enumerate(roi_list):
        print(f"\tROI {i+1}:\n"
              f"\t\t- (x_start, y_start): {roi.polygon[0]}\n"
              f"\t\t- (x_end, y_end): {roi.polygon[2]}")

    # ---------- TRACKER ----------
    DEVICE_STRING = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    DEVICE = torch.device(DEVICE_STRING)
    FP16 = True if (DEVICE_STRING != "cpu") else False
    TRACK_BUFFER = FRAME_RATE * 5  # 5 seconds

    tracker = BoTSORT(
        model_weights=Path('models/osnet_x0_25_msmt17.pt'),  # default: "osnet_x0_25_msmt17.pt"
        device=DEVICE,
        fp16=FP16,
        with_reid=True,
        track_buffer=TRACK_BUFFER,
        new_track_thresh=0.8,
        frame_rate=FRAME_RATE
    )

    print(f"Tracker information:\n"
          f"\t- type: BotSORT with reID\n"
          f"\t- device: {DEVICE_STRING}\n"
          f"\t- FP16: {FP16}\n")

    # ---------- DETECTOR ----------
    model = YOLO("models/yolov8m.pt", verbose=False)

    vid = cv2.VideoCapture(VIDEO_PATH)

    current_frame = 0

    while True:
        ret, img = vid.read()
        current_frame += FRAME_TO_SKIP
        vid.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        if ret:
            frame = img.copy()

            # input to tracker has to be N X (x, y, x, y, conf, cls)
            results = model(img, classes=0, device=DEVICE, verbose=False)

            # converting results in the format needed by the tracker
            dets = get_dets(results)

            # updating tracker
            tracks = tracker.update(dets, img)  # --> (x, y, x, y, id, conf, cls, ind)

            # print bboxes with their associated id, cls and conf
            if tracks.shape[0] != 0:
                # getting info from tracks
                xyxys = tracks[:, 0:4].astype('int')  # float64 to int
                ids = tracks[:, 4].astype('int')  # float64 to int
                confs = tracks[:, 5]
                clss = tracks[:, 6].astype('int')  # float64 to int
                inds = tracks[:, 7].astype('int')  # float64 to int

                for i, (xyxy, id, conf, cls) in enumerate(zip(xyxys, ids, confs, clss)):
                    # checks if it is a new id
                    person = people_det.get(id)
                    if person is None:
                        person = init_person(id)
                        people_det[id] = person

                    X, Y, X2, Y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                    person["last_frame"] = frame[Y:Y2, X:X2]

                    for j in range(i + 1, len(ids)):
                        xyxyA = xyxy
                        xyxyB = xyxys[j]
                        _, iou = checkIntersection(xyxyA, xyxyB)
                        personA = person
                        personB = people_det.get(ids[j])
                        if personB is None:
                            personB = init_person(ids[j])
                            people_det[ids[j]] = personB
                        if iou < 0.3:
                            personA["overlap"] = False
                            personB["overlap"] = False
                        else:
                            personA["overlap"] = True
                            personB["overlap"] = True
                            break

                    # updating the number of frames a person is in the scene
                    person["in_scene_count"] += 1

                    if person["in_scene_count"] % INFERENCE_RATE == 0:
                        if not person["overlap"]:
                            # getting cropped image from the bounding box to perform inference
                            X, Y, X2, Y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                            cropped_image = frame[Y:Y2, X:X2]
                            if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                                # adjusting image for inference
                                cropped_image = adjust_gamma(cropped_image, gamma=1.5)
                                infer_results = inference(cropped_image)
                                # updating person information with the latest predictions
                                update_person_info(person=person, infer_results=infer_results)

                                person["first_inference"] = True
                        else:
                            person["in_scene_count"] -= 1

                    # updating person information about region of interest behaviour
                    for i, roi in enumerate(roi_list):
                        # only if the person is in the roi
                        if is_person_in_roi(xyxy, roi):
                            person[f"roi{i+1}_persistence_time"] += 1 / FRAME_RATE
                            person[f"roi{i+1}_exit_count"] = 0

                            # updating number of people in roi in the current frame
                            if i == 0:
                                roi1_count += 1
                            else:
                                roi2_count += 1

                            # updating current and last state
                            person[f"roi{i+1}_last_state"] = person[f"roi{i+1}_current_state"]
                            person[f"roi{i+1}_current_state"] = True

                            if person[f"roi{i+1}_current_state"] and person[f"roi{i+1}_last_state"] != person[f"roi{i+1}_current_state"]:
                                person[f"roi{i+1}_passages"] += 1
                                if i == 0:
                                    roi1_passages += 1
                                else:
                                    roi2_passages += 1
                        else:
                            person[f"roi{i+1}_exit_count"] += 1
                            if person[f"roi{i+1}_exit_count"] == EXIT_COUNT:
                                person[f"roi{i+1}_last_state"] = person[f"roi{i+1}_current_state"]
                                person[f"roi{i+1}_current_state"] = False

                    # checks if the person is in the roi in order to change its bounding box color
                    color = ROI1_COLOR if person["roi1_current_state"] else ROI2_COLOR \
                        if person["roi2_current_state"] else BBOX_COLOR

                    # getting labels for information drawing
                    id_label = str(person["id"])
                    gender_label = person['gender'] if person['gender'] is not None else "?"
                    bag_label = "?" if person['bag'] is None else "Bag" if person['bag'] else "No Bag"
                    hat_label = "?" if person['hat'] is None else "Hat" if person['hat'] else "No Hat"
                    upper_label = person['upper_color'] if person['upper_color'] is not None else "?"
                    lower_label = person['lower_color'] if person['lower_color'] is not None else "?"

                    # top left person bounding box, id
                    x_id, y_id = xyxy[0] + DX, xyxy[1] + DY
                    x2_id, y2_id = x_id + 30, y_id + 30
                    img = cv2.rectangle(
                        img,
                        (x_id, y_id),
                        (x2_id, y2_id),
                        WHITE,
                        -1  # filling
                    )

                    x_pos = int((x_id+x2_id)/2 - DX) if person["id"] < 10 else int((x_id+x2_id)/2 - 2*DX)
                    img = cv2.putText(
                        img,
                        id_label,
                        (x_pos, int((y_id+y2_id)/2) + DY),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONTSCALE,
                        color,
                        THICKNESS
                    )

                    # bottom person bounding box, all info
                    text = (f"Gender: {gender_label[0].upper()}\n"
                            f"{bag_label} {hat_label}\n"
                            f"U-L: {upper_label.capitalize()}-{lower_label.capitalize()}"
                            )

                    # drawing person bounding box
                    img = cv2.rectangle(
                        img,
                        (xyxy[0], xyxy[1]),
                        (xyxy[2], xyxy[3]),
                        color,
                        THICKNESS
                    )

                    x_id, y_id = xyxy[0] + DX, xyxy[3] + DY
                    x2_id, y2_id = x_id + 180, y_id + 60
                    img = cv2.rectangle(
                        img,
                        (x_id, y_id),
                        (x2_id, y2_id),
                        WHITE,
                        -1  # filling
                    )

                    # drawing person information near its bounding box
                    x, y0, dy = xyxy[0] + 2*DX, xyxy[3] + 2*DY + 8, 20
                    for i, line in enumerate(text.split('\n')):
                        y = y0 + i * dy

                        cv2.putText(
                            img,
                            line,
                            (x, y),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.6,
                            ROI_COLOR,
                            1
                        )

            # drawing rois on image
            for i, roi in enumerate(roi_list):
                if i == 0:
                    roi_text = "1"
                else:
                    roi_text = "2"
                cv2.putText(
                    img,
                    roi_text,
                    (roi.polygon[0][0] + 15, roi.polygon[0][1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    ROI_COLOR,
                    ROI_THICKNESS
                )
                img = cv2.rectangle(
                    img,
                    roi.polygon[0],
                    roi.polygon[2],
                    ROI_COLOR,
                    ROI_THICKNESS
                )

            # drawing general information on top left screen
            # top left person bounding box, id
            img = cv2.rectangle(
                img,
                (0, 0),
                (400, 165),
                WHITE,
                -1  # filling
            )

            total_persons = tracks.shape[0]
            text = (f"People in ROI: {roi1_count + roi2_count}\n"
                    f"Total persons: {total_persons}\n"
                    f"Passages in ROI 1: {roi1_passages}\n"
                    f"Passages in ROI 2: {roi2_passages}")

            x, y0, dy = 20, 40, 30
            for i, line in enumerate(text.split('\n')):
                y = y0 + i * dy

                cv2.putText(
                    img,
                    line,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    ROI_COLOR,
                    2
                )

            # show image with bboxes, ids, classes and confidences
            cv2.imshow('frame', img)

            # resetting number of people in rois
            roi1_count = 0
            roi2_count = 0

            # break on pressing q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # ----------- FINALIZATION ----------
    vid.release()
    cv2.destroyAllWindows()

    # making inference on people on which was not made inference
    for person in people_det.values():
        if person["first_inference"] is False:
            print("Performing inference on person with ID", person["id"])
            cropped_image = person["last_frame"]
            if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                # adjusting image for inference
                cropped_image = adjust_gamma(cropped_image, gamma=1.5)
                infer_results = inference(cropped_image)

                # updating person information with the latest predictions
                update_person_info(person=person, infer_results=infer_results)

    # ----------- OUTPUT SAVING ----------
    #  saving people information in json file
    save_output(OUTPUT_FILENAME, people_det)

if __name__ == "__main__":
    main()