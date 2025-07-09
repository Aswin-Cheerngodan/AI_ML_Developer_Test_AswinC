from typing import Optional
from fastapi import UploadFile
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import shutil
import uuid
import os

class DataIngestion:
    """Class to handle data ingestion for chart trend classification."""
    def __init__(self, upload_dir: Path = Path(r"app/static/uploads")):
        """ Initialize with image directory.
        
        Args:
            upload_dir (Path): Directory to store uploaded images. Defaults to r"app/static/uploads"
        """
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    
    def ingest_image(self, file: UploadFile):
        """Accept, save and validate the image.
        
        Args:
            file (UploadFile): Uploaded image from FastAPI.

        Retruns:
            file : Validated image file or None.
        """
        try:
            # Validate file type
            if not file.content_type.startswith("image/"):
                return None

            # Save the file
            unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = self.upload_dir / unique_filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            return file_path
        except Exception as e:
            
            return None
        


class DataPreprocessor:
    """Class to preprocess the data for the model input."""
    def __init__(self, file_path: Path):
        """ Initialize with file path of the image for preprocessing.

        Args:
            file_path (str): path of the image.
        
        """
        self.file_path = file_path

    
    def preprocess(self, image_size=(224, 224)) -> Optional[np.ndarray]:
        """ Preprocess the image for CNN input.

        Returns:
            Optional[np.ndarray]: Preprocessed image array.shape:(1, height, width, channels), or None if preprocessing fails.
        """
        try:
            # Open and resize image
            img = cv2.imread(self.file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            resized_image = cv2.resize(img, image_size)
            img_normalized = resized_image / 255.0
            img_array = np.expand_dims(img_normalized, axis=0)
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
            return img_array
        except Exception as e:
            print(e)
            return None


class ChartClassifier:
    """Class handles chart classification."""
    def __init__(self, model_path: Path = Path(r"artifacts/models/chart_trend_model.h5")):
        """Intialize with path of the model.
        
        Args:
            model_path (Path): Path to the chart trend classifier model. Defaults to "stock-chart-trend-classification/artifacts/models/chart_trend_model.h5"
        """
        self.model_path = model_path
        self.classes = ["Downtrend", "Uptrend"]


    def _load_model(self) -> Model:
        """Loading the CNN model for chart trend classification from model_path

        Returns (Model): Loaded CNN model or None if loading fails.
        """
        try:
            model = load_model(self.model_path)
            return model
        except Exception as e:
            print(e)
            return None
        
    def classify(self, data: np.ndarray) -> Optional[str]:
        """Classifies the input data into downtrend and uptrend.
        Args:
            data (np.ndarray): Numpy array with shape (1, 224, 224, 3) for matching the input shape of the model.

        Returns:
            Optional (str): Downtrend or Uptrend. None if prediction fails.
        """
        try:
            model = self._load_model()
            pred = model.predict(data)
            if not pred.any():
                return None
            trend = self.classes[np.argmax(pred)]
            return trend
        except Exception as e:
            print(e)
            return None
        
class TrendClassifier:
    """Executes the prediction pipeline for chart trend classification."""
    def __init__(self, file:UploadFile) -> None:
        """Initialize with the file to be used for classification.
        Args:
            file (img): Image file for the classification.
        """
        self.file = file

    def classify(self) -> Optional[str]:
        """Executes the entire pipeline.
        Returns:
            Optional[str]: Classified class (Downtrend or Uptrend). None if classification fails.
        """
        try:
            data_ingestor = DataIngestion()
            file_path = data_ingestor.ingest_image(self.file)

            preprocessor = DataPreprocessor(file_path)
            img_array = preprocessor.preprocess()

            classifier = ChartClassifier()
            trend = classifier.classify(img_array)
             
            return trend
        except Exception as e:
            print(e)
            return None
        


# if __name__=="__main__":
#     classifier = TrendClassifier(Path(r"stock-chart-trend-classification/app/static/uploads/Screenshot 2025-04-22 201616.png"))
#     trend = classifier.classify()
#     trend = str(trend)
#     print(trend)

        