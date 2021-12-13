import torch
from crop import crop, draw_axis, draw_axis_ege
import cv2
import albumentations as A
import albumentations.pytorch as Ap
import MyModels as models
import MyModels_ege as models_ege

def get_grad(img,SIZE = 112):
    path = ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.WHENet()
    if SIZE == 112:
        path = "snp/UE_112_NashDetecter_5_L1_300W_LP_262-27.3278.tar"
        path = "snp/UE+300W_LP_112_NashDetecter_L1_300W_LP_279-19.6356.tar"

    elif SIZE == 144:
        path = "snp/UE_144_NashDetecter_5_L1_300W_LP_195-25.0709.tar"
        path = "snp/UE+300W_LP_144_NashDetecter_L1_300W_LP_240-18.0052.tar"

    elif SIZE == 176:
        path = "snp/UE_176_NashDetecter_5_L1_300W_LP_278-23.6223.tar"
        path = "snp/UE+300W_LP_176_NashDetecter_L1_300W_LP_214-17.2976.tar"

    elif SIZE == 208:
        path = "snp/UE_208_NashDetecter_5_L1_300W_LP_296-23.2967.tar"
        path = "snp/UE+300W_LP_208_NashDetecter_L1_300W_LP_153-16.7829.tar"

    elif SIZE == 224:
        path = "snp/UE_224_NashDetecter_4_L1_300W_LP_41-23.6047.tar"
        path = "snp/UE+300W_LP_224_NashDetecter_L1_300W_LP_194-16.4275.tar"
    else:
        return "Размер может быть 112, 144, 176, 208, 224"

    val_transforms = A.Compose([
            A.Resize(SIZE, SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            Ap.transforms.ToTensorV2()
            ])

    model = model.to(device)
    weights = torch.load(path, map_location='cpu')
    model.load_state_dict(weights['state_dict'], strict=True)
    model = model.eval()

    path_ege = "snp/ege/UE_50%2A100_2Dat_99-54-54_L2_785-95.3554.tar"
    model_ege = models_ege.WHENet()
    SIZE_ege = 50
    model_ege = model_ege.to(device)
    weights = torch.load(path_ege, map_location='cpu')
    model_ege.load_state_dict(weights['state_dict'], strict=True)
    model_ege = model_ege.eval()
    val_transforms_ege = A.Compose([
        A.Resize(SIZE_ege, SIZE_ege * 2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        Ap.transforms.ToTensorV2()
    ])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    grad = []
    cpop, b = crop(img)
    frame = img
    if cpop:
        for i in range(len(cpop)):
            try:
                fr = frame[int(cpop[i][1]):int(cpop[i][3]), int(cpop[i][0]):int(cpop[i][2])]
                img_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                inputs = val_transforms(image=img_rgb)["image"].unsqueeze(0)
                inputs = inputs.to(device)
                classification_yaw, classification_pitch, classification_roll, regression_yaw, regression_pitch, regression_roll = model(inputs)
                img = draw_axis(img, regression_yaw, regression_pitch, regression_roll,
                                int(cpop[i][0] + (cpop[i][2]-cpop[i][0])/2),
                                int(cpop[i][1] + (cpop[i][3]-cpop[i][1])/2),)
                img = cv2.putText(img, str(int(regression_yaw.squeeze(0).cpu().detach().numpy())), (10 + i * 100, 50), font, fontScale, color, thickness, cv2.LINE_AA)
                img = cv2.putText(img, str(int(regression_pitch.squeeze(0).cpu().detach().numpy())), (10 + i * 100, 100), font, fontScale, color, thickness, cv2.LINE_AA)
                img = cv2.putText(img, str(int(regression_roll.squeeze(0).cpu().detach().numpy())), (10 + i * 100, 150), font, fontScale, color, thickness, cv2.LINE_AA)

                cv2.rectangle(img, (cpop[i][0], cpop[i][1]), (cpop[i][2], cpop[i][3]), (0, 0, 255), 2)
                grad.append([int(regression_yaw.squeeze(0).cpu().detach().numpy()),int(regression_pitch.squeeze(0).cpu().detach().numpy()),int(regression_roll.squeeze(0).cpu().detach().numpy())])

                fr = frame[int(cpop[i][9]):int(cpop[i][11]), int(cpop[i][8]):int(cpop[i][10])]
                img_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                inputs = val_transforms_ege(image=img_rgb)["image"].unsqueeze(0)
                inputs = inputs.to(device)
                classification_yaw, classification_pitch, classification_roll, regression_yaw, regression_pitch, regression_roll = model_ege(inputs)
                img = draw_axis_ege(img, regression_yaw, regression_pitch, regression_roll,
                                    int(cpop[i][8] + (cpop[i][10] - cpop[i][8]) / 2),
                                    int(cpop[i][9] + (cpop[i][11] - cpop[i][9]) / 2))
                img = cv2.putText(img, str(int(regression_yaw.squeeze(0).cpu().detach().numpy())),
                                  (cpop[i][8], cpop[i][9]), 1, 1, (0, 0, 255), thickness, cv2.LINE_AA)
                img = cv2.putText(img, str(int(regression_pitch.squeeze(0).cpu().detach().numpy())),
                                  (cpop[i][8], cpop[i][9] + 20), 1, 1, (0, 0, 255), thickness, cv2.LINE_AA)

                fr = frame[int(cpop[i][13]):int(cpop[i][15]), int(cpop[i][12]):int(cpop[i][14])]
                img_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                inputs = val_transforms_ege(image=img_rgb)["image"].unsqueeze(0)
                inputs = inputs.to(device)
                classification_yaw, classification_pitch, classification_roll, regression_yaw, regression_pitch, regression_roll = model_ege(inputs)
                img = draw_axis_ege(img, regression_yaw, regression_pitch, regression_roll,
                                    int(cpop[i][12] + (cpop[i][14] - cpop[i][12]) / 2),
                                    int(cpop[i][13] + (cpop[i][15] - cpop[i][13]) / 2))
                img = cv2.putText(img, str(int(regression_yaw.squeeze(0).cpu().detach().numpy())),
                                  (cpop[i][14], cpop[i][13]), 1, 1, (0, 0, 255), thickness, cv2.LINE_AA)
                img = cv2.putText(img, str(int(regression_pitch.squeeze(0).cpu().detach().numpy())),
                                  (cpop[i][14], cpop[i][13] + 20), 1, 1, (0, 0, 255), thickness, cv2.LINE_AA)
                cv2.rectangle(img, (cpop[i][8], cpop[i][9]), (cpop[i][10], cpop[i][11]), (0, 0, 255), 2)
                cv2.rectangle(img, (cpop[i][12], cpop[i][13]), (cpop[i][14], cpop[i][15]), (0, 0, 255), 2)
            except:
                return "Что-то пошло не так"
    return grad,img

image = cv2.imread("masshtabirovanie-biznesa-franchajzing.jpg")
grad,img = get_grad(image,SIZE = 112)
print(grad)
cv2.imshow('Amanda', img)
cv2.waitKey(0)
