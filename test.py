import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model

from utils.metrics import ComputeMetrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", help="Test Images")
    parser.add_argument("--labels", help="Test Labels")
    parser.add_argument("--model_path", help="model path")
    parser.add_argument("--save_output", help="save output or not") 
    args = parser.parse_args()
    
    # compute metrics module
    compute_metric = ComputeMetrics()
    # Load model
    model = load_model(args.model_path)
    
    # Read test data
    test_data = glob.glob(args.images)
    test_mask = glob.glob(args.labels)
    
    
    iou_scores = []
    for i in tqdm(range(len(test_data))):
    
        img = cv2.imread(test_data[i])
        img_resized = cv2.resize(img, (176, 256))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img = gray[np.newaxis,:,:,np.newaxis]
        predict_mask = model.predict(img)
        volume = predict_mask[0,:,:,:]
        v_max = np.max(volume)
        v_min = np.min(volume)
        volume_norm = (volume - v_min) / (v_max - v_min)
        volume_norm = (volume_norm * 255).astype("int")
        
        if args.save_output()
            cv2.imwrite(os.path.join("prediction", test_data[i].split("/")[-1]), volume_norm)
            
        maskP = cv2.imread(test_mask[i])
        img_resized = cv2.resize(maskP, (176, 256))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        iou_pred = compute_metric.iou(volume_norm, gray[:,:,np.newaxis])
        iou_scores.append(iou_pred)
        
     print("Evaluation done.")
     print("mIOU is:", np.mean(np.array(iou_scores)))
     
if __name__ == "__main__":
    main()