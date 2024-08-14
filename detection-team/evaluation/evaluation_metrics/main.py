from autodistill_grounding_dino import GroundingDINO
from autodistill_yolov8 import YOLOv8
from autodistill.detection import CaptionOntology
from ConcreteEvaluationMetrics import ConcreteEvaluationMetrics
from prediction_evaluator import PredictionEvaluator

# Initialize models
predictor = GroundingDINO(ontology=CaptionOntology({"shipping container": "container"}))

# Instantiate the concrete subclass
metrics = ConcreteEvaluationMetrics(
    predictor=predictor,
    images_path='./images',
    labels_path='/notebooks/labels/ground_truth.csv'  # Correct path to the CSV file
)

# Initialize evaluator
evaluator = PredictionEvaluator(metrics=metrics, image_folder='./images', label_folder='/notebooks/labels/ground_truth.csv')

# Evaluate and save metrics
evaluator.evaluate_and_save_metrics(metrics_csv_path='metrics_results.csv', recall_csv_path='recall_scores.csv')
