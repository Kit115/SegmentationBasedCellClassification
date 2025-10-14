from models import NucleusSegmenter, CellClassifier
import torch

from data import EvaluationDataset


@torch.no_grad()
def run_segmentation(segmentation_model, image, method="connected_components", threshold=0.5, nms_radius=16):
    image = FT.resize(image, (224, 224))
    linspace = np.linspace(0.0, 1.0, 224)
    points = np.zeros((224, 224, 2))

    for y in range(224):
        for x in range(224):
            points[y, x, :] = [linspace[x], linspace[y]]
                
    segmentation_map = segmentation_model(
        torch.from_numpy(points).reshape(1, -1, 2).float(), 
        image.unsqueeze(0),
        to_prob=True
    )
    segmentation_map = segmentation_map.reshape(224, 224)
    segmentation_map[segmentation_map < threshold] = threshold
    
    if method == "connected_components":
        bboxes, keypoints = heatmap_to_bboxes(segmentation_map, threshold=threshold)
        keypoints = torch.tensor(keypoints)
        extra = {"bboxes": bboxes}

    elif method == "maxima_extraction":
        keypoints = find_heatmap_peaks(segmentation_map, min_confidence=threshold, nms_radius=12, sigma=2.0)
        extra = None
    else:
        raise
    return segmentation_map, keypoints, extra

@torch.no_grad()
def run_classification(classification_model, image, keypoints):
    numpy_image = image.permute(1, 2, 0).numpy()
    H, W, _ = numpy_image.shape

    cropped_images = []
    
    for keypoint in keypoints:
        x, y = keypoint
        x, y = int(x * W), int(y * H)

        cropped_images.append(crop_image(numpy_image, (x, y), 224))

    cropped_images = torch.from_numpy(np.stack(cropped_images, 0)).permute(0, 3, 1, 2)
    predictions = classification_model(cropped_images).argmax(-1)
    return predictions.tolist()

def find_mask_match(mask_list, x, y):
    for mask in mask_list:
        x_rel, y_rel = int(x - mask["min_x"]), int(y - mask["min_y"])
        mask_width = mask["max_x"] - mask["min_x"]
        mask_height = mask["max_y"] - mask["min_y"]
        
        if not (0 <= x_rel < mask_width):
            continue
        if not (0 <= y_rel < mask_height):
            continue
    
        numpy_mask = base64_mask_to_numpy(mask["base64_mask"])[:, :, 0]
        if numpy_mask[y_rel, x_rel] != 0.:
            return mask
     
    return None

def evaluate_predictions(mask_list, keypoints, class_predictions):
    mask_list = deepcopy(mask_list)

    correct_classifications = 0
    incorrect_classifications = 0

    true_positive_detections = 0
    false_positive_detections = 0
    false_negative_detections = 0
    
    for keypoint, class_prediction in zip(keypoints, class_predictions):
        x, y = keypoint
        x, y = int(x), int(y)
        
        mask_match = find_mask_match(mask_list, x, y)
        if mask_match is not None:
            mask_list.remove(mask_match)

            true_positive_detections  += 1
            if mask_match['label'] != -1:
                if class_prediction == mask_match['label']:
                    correct_classifications   += 1
                else:
                    incorrect_classifications += 1

        else:
            false_positive_detections += 1

    false_negative_detections = len(mask_list)
    
    return {
        "correct_classifications": correct_classifications,
        "incorrect_classifications": incorrect_classifications,
        "true_positive_detections": true_positive_detections,
        "false_positive_detections": false_positive_detections,
        "false_negative_detections": false_negative_detections
    }


def print_metrics(dataset_stats):
    c_c = dataset_stats['correct_classifications']
    i_c = dataset_stats['incorrect_classifications']

    tp_d = dataset_stats['true_positive_detections']
    fp_d = dataset_stats['false_positive_detections']
    fn_d = dataset_stats['false_negative_detections']

    print("Detection Metrics:")
    print(f"Recall:    {(100. * (tp_d/(tp_d + fn_d))):.2f}%")
    print(f"Precision: {(100. * (tp_d/(tp_d + fp_d))):.2f}%")
    print("---")
    print("Classification Metrics:")
    print(f"Accuracy:  {100. * (c_c / (c_c + i_c)):.2f}%")
   


def main(args):
    print(f"WARNING: Currently using the {args.split} split for evaluation.")

    use_cuda    = args.use_cuda and torch.cuda.is_available()
    device      = "cuda:0" if use_cuda else "cpu"
    
    segmentation_model = NucleusSegmenter().to(device)
    segmentation_model.load_state_dict(torch.load(
        "checkpoints/segmentation_checkpoint.pth", 
        weights_only=False, 
        map_location=device
    )["model"])
    segmentation_model.eval()
    
    classification_model = CellClassifier().to(device)
    classification_model.load_state_dict(torch.load(
        "checkpoints/classification_model_checkpoint.pth", 
        weights_only=False,
        map_location=device
    )["model"])
    classification_model.eval()


    dataset_stats = {
        "correct_classifications": 0,
        "incorrect_classifications": 0,
        "true_positive_detections": 0,
        "false_positive_detections": 0,
        "false_negative_detections": 0
    }

    for i in tqdm(range(len(dataset))):
        image, _, mask_list = dataset[i]
        segmentation_map, keypoints, _  = run_segmentation(segmentation_model, image, threshold=args.detection_threshold, method=args.keypoint_extraction_method)
        class_predictions               = run_classification(classification_model, image, keypoints)
        scaled_keypoints                = scale_keypoints_to_image(image, keypoints)
        
        sample_stats =  evaluate_predictions(
            mask_list, 
            scaled_keypoints, 
            class_predictions
        )
        for key in sample_stats:
            dataset_stats[key] += sample_stats[key]

    print_metrics(dataset_stats) 
    


if __name__ == "__main__":
    import Argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--split',                      type=str,   default="test",                 choices=["train", "validation", "test"])
    parser.add_argument('--keypoint-extraction-method', type=str,   default="maxima_extraction",    choices=["maxima_extraction", "connected_components"])
    parser.add_argument('--detection-threshold',        type=float, default=0.95)

    parser.add_argument('--cuda',    dest="use_cuda", action='store_true')
    parser.add_argument('--no-cuda', dest='use_cuda', action='store_false')
    parser.set_defaults(use_cuda=True)

    args = parser.parse_args()
    main(args)
