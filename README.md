# CNN Image Classification with Transfer Learning

## CNN Class Exercise by Maxwell Ernst - 03/07/2023

### Overview

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for image classification using transfer learning. The goal is to classify images of cats and dogs by leveraging a pre-trained MobileNetV2 model. The project covers various aspects, including data handling, model building, fine-tuning, and evaluation.

### Project Structure

- **Notebook:** `CNN_Class_Exercise.ipynb`
- **Libraries:** TensorFlow, Matplotlib, NumPy
- **Data:** Cats and dogs dataset from [Google's tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning)
- **Objective:** Implement a CNN for image classification, utilizing transfer learning with a pre-trained model.

### Steps

1. **Data Handling:**
   - Load and preprocess the cats and dogs dataset.
   - Use TensorFlow's image dataset utilities.
   - Apply data augmentation for improved model generalization.

2. **Model Building:**
   - Use MobileNetV2 as a pre-trained model for feature extraction.
   - Freeze convolutional layers to retain pre-trained weights.
   - Add a custom classification head for the specific task.

3. **Training:**
   - Train the model on the dataset for an initial set of epochs.
   - Plot accuracy and loss versus epoch number to visualize training progress.

4. **Fine-Tuning:**
   - Unfreeze selected layers of the pre-trained model.
   - Retrain the model with a lower learning rate for fine-tuning.

5. **Evaluation:**
   - Evaluate the model on a test dataset.
   - Generate a confusion matrix for detailed performance analysis.

6. **Results:**
   - Achieved a test accuracy of approximately 97.9%.
   - Confusion matrix provides insights into model performance.

### Conclusion

This project showcases the practical application of transfer learning in image classification. By leveraging a pre-trained model, we achieved high accuracy in distinguishing between cats and dogs. The code is well-documented, and visualizations help in understanding the training process and model performance.

Feel free to explore the Jupyter notebook for a detailed walkthrough of the project.

### Instructions

1. Open `CNN_Class_Exercise.ipynb` in a Jupyter environment.
2. Run the cells sequentially to execute the code.
3. Analyze the visualizations and results to understand the model's performance.

### Dependencies

- TensorFlow
- Matplotlib
- NumPy

### Author

Maxwell Ernst

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
