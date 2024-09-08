
# CS 109b: Deep Learning - Harvard University

This repository contains course materials, assignments, and solutions for the **Advanced Topics in Data Science (CS109b)** course taught by Prof. Pavlos Protopapas (SEAS) and Alex Young (Statistics) at Harvard University. This course is the second part of a year-long introduction to data science, building upon the foundation laid in CS109a. 

The course covers advanced data science techniques, including **unsupervised learning, clustering, Bayesian inference, hierarchical Bayesian modeling, neural networks (fully connected, convolutional, autoencoders, and recurrent), natural language processing (NLP) and text analysis, transformers, and generative adversarial networks (GANs)**. Python will be the primary programming language used for all assignments and exercises.

## Course Overview

**Course Title**: Advanced Topics in Data Science  
**Instructors**: Prof. Pavlos Protopapas (SEAS) & Alex Young (Statistics)  
**Institution**: Harvard University

### Assignment 1: Clustering (cs109b_hw1.ipynb)

- **Problem Statement:**  
  The goal of this assignment is to explore different clustering techniques such as k-means and DBSCAN to identify clusters in a dataset of audio features from the Free Music Archive (FMA). The assignment focuses on applying and evaluating clustering algorithms, using metrics like inertia, silhouette scores, and the gap statistic to determine the optimal number of clusters.

- **Dataset:**  
  The dataset contains 82 features for each music track, including:
  - 8 interpretable features such as acousticness, danceability, energy, and tempo.
  - 74 abstract audio features computed using the `librosa` library and averaged over the track runtime.
  The goal is to cluster the music tracks into groups based on their features.

- **Approach:**  
  The assignment is divided into three main parts:
  
  **Part 1: Clustering with k-means**  
  - **Data Standardization:** The data was standardized to ensure all features have a mean of 0 and a standard deviation of 1.
  - **K-Means Clustering:** K-means was applied with 12 clusters and visualized using silhouette plots and PCA projection. The silhouette score was used to evaluate the clustering quality, revealing suboptimal cluster separation.
  
  **Part 2: Other Ks**  
  - **Inertia and Elbow Method:** The elbow method was used to evaluate k-means for different values of k (from 1 to 15). The optimal k was identified by observing where inertia showed diminishing returns.
  - **Silhouette Score:** Silhouette scores were calculated for each k, further helping identify the best number of clusters.
  - **Gap Statistic:** The gap statistic was used to find the optimal number of clusters by comparing the within-cluster dispersion with a null reference distribution. The "slack" rule was applied to determine the best k.
  - **Best k Selection:** After comparing the elbow method, silhouette score, and gap statistic results, the best k was determined to be 2.
  
  **Part 3: DBSCAN**  
  - **DBSCAN Clustering:** DBSCAN was applied to the dataset, with the epsilon parameter tuned using the knee plot method. The number of clusters found by DBSCAN was analyzed, revealing one dominant cluster and several smaller clusters.
  - **Comparison with K-Means:** The genre makeup of clusters found by both DBSCAN and K-Means was compared, with K-Means showing more balanced and interpretable results.

- **Techniques Used:**
  - Data Standardization using `StandardScaler`
  - K-Means Clustering using `KMeans`
  - Inertia and Elbow Method for evaluating cluster numbers
  - Silhouette Analysis using `silhouette_score`
  - Gap Statistic using `OptimalK`
  - DBSCAN Clustering using `DBSCAN`
  - Visualization using Silhouette Plots, PCA, and Genre Makeup Analysis


### Assignment 2: Bayesian Analysis (cs109b_hw2.ipynb)

- **Problem Statement:**  
  The goal of this assignment is to apply Bayesian analysis to model contraceptive usage among Bangladeshi women, using logistic regression models with varying intercepts and coefficients. The assignment involves fitting models using PyMC, diagnosing convergence, and evaluating model performance using both simulated and real data.

- **Dataset:**  
  The dataset contains information on contraceptive use among 1,934 women, with the following features:
  - `district`: Code identifying the district of residence (60 districts total).
  - `urban`: Binary variable indicating if the woman lives in an urban area.
  - `living.children`: Number of living children.
  - `age_mean`: Age of the woman (centered around the mean).
  - `contraceptive_use`: Binary outcome indicating contraceptive usage (1 = yes, 0 = no).

- **Approach:**  
  The assignment is divided into four parts:

  **Part 1: Varying-Intercept Model (Simulated Data)**  
  - Simulated response values were generated using specified parameters to test if PyMC could recover the hidden values through Bayesian inference.
  - The varying-intercept model was constructed, with intercepts varying by district, and MCMC sampling was used to fit the model. Convergence was evaluated through trace plots and R-hat values.

  **Part 2: Varying-Intercept Model (Real Data)**  
  - The varying-intercept model was applied to the real contraceptive usage data. District-level intercepts were estimated, and the highest and lowest base rates of contraceptive usage across districts were identified.

  **Part 3: Varying-Coefficients Model**  
  - A more complex varying-coefficients model was fitted, where both the intercepts and the coefficients for urban, living children, and age varied by district. This model was evaluated for convergence and the distribution of posterior means was analyzed.

  **Part 4: Prediction & Model Selection**  
  - The predictions from the varying-intercept and varying-coefficients models were compared in terms of accuracy and the percentage of observations predicted as contraceptive users. A naive baseline model (predicting the most frequent outcome) was also evaluated, and the models were compared to determine the best fit.

- **Techniques Used:**
  - Bayesian Logistic Regression using `PyMC`
  - MCMC sampling for parameter estimation
  - Posterior inference and convergence diagnostics using `ArviZ`
  - Varying-Intercept and Varying-Coefficients Models
  - Model evaluation using accuracy, posterior predictive checks, and R-hat values


### Assignment 3: Artificial Neural Networks, Model Interpretation, and Regularization (cs109b_hw3.ipynb)

- **Problem Statement:**  
  This assignment focuses on artificial neural networks (ANNs) and model interpretation. The primary tasks are to build, evaluate, and interpret ANN models to predict flight delays. Additionally, students will participate in a Kaggle competition using the Kannada MNIST dataset for handwritten digit classification, focusing on regularizing an overfit model.

- **Dataset:**  
  **Part 1**: The dataset consists of flight information with variables such as scheduled arrival and departure times, flight distance, and whether the flight was delayed by 15 minutes or more (`DELAY_OR_NOT`).  
  **Part 2**: The dataset consists of images of handwritten digits in the Kannada language. Each image is a 28x28 grayscale image, with class labels for digits 0 and 1. The task is to predict the handwritten digits using regularized neural networks.

- **Approach:**  
  The assignment is divided into two main parts:

  **Part 1: Model Interpretation and Predictive Intervals in ANNs**  
  - **Data Preprocessing**: The flight dataset was cleaned, and missing values were handled. Categorical variables such as origin and destination airports were one-hot encoded. The dataset was then split into training and test sets.
  - **Neural Network Model**: An artificial neural network was built with two hidden layers and trained to predict flight delays. Training and validation accuracy were monitored to evaluate performance. A bootstrap method was used to create multiple training sets, and separate neural networks were fitted to each bootstrapped set to estimate uncertainty in predictions.
  - **Abstain Model**: A bagging model was developed to abstain from making predictions if there was too much uncertainty. The accuracy of the abstain model was evaluated across different confidence thresholds.

  **Part 2: Kannada MNIST Kaggle Competition**  
  - **Overfit Model**: A fully-connected neural network with three hidden layers was built and overfitted to the Kannada MNIST training data. The overfitting was demonstrated through the discrepancy between high training accuracy and low validation accuracy.
  - **Regularization**: Various regularization techniques were implemented, including dropout, L2 regularization, and data augmentation. The goal was to prevent overfitting and improve generalization on the test set.
  - **Kaggle Competition**: The regularized model was used to generate predictions on the test set, which were submitted to the Kaggle competition for leaderboard scoring.

- **Techniques Used:**
  - **Artificial Neural Networks (ANNs)**: A feed-forward network was built using TensorFlow/Keras to predict flight delays and classify handwritten digits.
  - **Bootstrap Aggregation (Bagging)**: Bootstrapping was used to generate prediction intervals and assess model uncertainty.
  - **Data Augmentation**: Techniques such as rotation, shifting, and zooming were applied to the training data to prevent overfitting in the Kannada MNIST task.
  - **Regularization**: Dropout and L2 regularization were used to prevent overfitting in the neural network models.

### Assignment 4: Convolutional Neural Networks (cs109b_hw4.ipynb)

- **Problem Statement:**  
  This assignment explores the use of convolutional neural networks (CNNs) for image classification and regression tasks. The tasks involve building a CNN model for object classification using the CIFAR-10 dataset and training another CNN to predict image rotations using the CelebA dataset.

- **Dataset:**  
  **Part 1**: The CIFAR-10 dataset contains 60,000 32x32 colored images of 10 object classes, with 50,000 images for training and 10,000 for testing.
  **Part 2**: The CelebA dataset contains celebrity faces. Each image is rotated, and the task is to predict the angle by which an image is rotated and then correct the orientation.

- **Approach:**  
  The assignment is divided into two main parts:

  **Part 1: Object Classification with CNNs**  
  - **Data Preprocessing**: The CIFAR-10 dataset is loaded, normalized, and split into training and test sets.
  - **CNN Architecture**: A convolutional neural network is constructed using layers such as Conv2D, MaxPooling2D, Dense, Dropout, and Flatten. The model is designed to classify the CIFAR-10 images.
  - **Model Training**: The model is trained using a validation split, and the validation and test accuracies are reported. The model is trained for multiple epochs to achieve a test accuracy of at least 70%.
  - **Parameter Growth Analysis**: The growth of model parameters is analyzed as the number of filters in the convolutional layers increases. A plot is generated to show the relationship between the number of filters and the total number of parameters.
  - **Results**: The training and validation accuracies are plotted, and the performance of the model is evaluated based on the test set.

  **Part 2: Image Orientation Regression with CNNs**  
  - **Data Preparation**: The CelebA dataset is loaded, and images are rotated by random degrees between -60° and 60°. A TensorFlow Dataset object is created, which allows for pipelining transformations such as resizing, rotating, and normalizing images.
  - **CNN Architecture**: A convolutional neural network is built to predict the degree of rotation of an image. The network consists of several Conv2D, MaxPooling2D, and Dense layers.
  - **Model Training**: The model is trained on the rotated CelebA images to predict the angle by which an image has been rotated. The test mean squared error (MSE) is reported, and the goal is to achieve a test MSE of less than 9.
  - **Image Correction**: The model is used to predict the rotation of test images, and the images are corrected based on the predicted rotation. The corrected images are displayed alongside the original and actual rotation values.
  - **Model Saving and Loading**: The model's weights are saved and later reloaded to demonstrate how to resume training or use the model for inference.
  - **Results**: The model's performance is evaluated on test images, and a custom image is corrected using the trained model.

- **Techniques Used:**
  - **Convolutional Neural Networks (CNNs)**: CNNs are used for both image classification and regression tasks. The architecture includes convolutional, pooling, dropout, and dense layers.
  - **Data Augmentation**: Random rotations, resizing, and normalization are applied to the CelebA dataset to simulate different orientations and improve the model's generalization.
  - **Model Saving and Loading**: The model's weights are saved and reloaded to demonstrate how to resume training or use the model for prediction.
  - **Visualization**: The model's predictions are visualized by displaying the original, predicted, and corrected images with their respective rotation angles.


### Assignment 5: Language Modeling & Recurrent Neural Networks (cs109b_hw5.ipynb)

- **Problem Statement:**  
  This assignment involves building two distinct models using Recurrent Neural Networks (RNNs) for language tasks: text generation and Named Entity Recognition (NER). The tasks require designing models to generate text at the character level and recognize entities in sentences.

- **Dataset:**
  - **Part 1 (Text Generation)**: The dataset contains text from Edward Lear, a 19th-century English writer known for his whimsical prose and poetry.
  - **Part 2 (Named Entity Recognition)**: The NER dataset consists of sentences, each word tagged with part-of-speech and named entity information. The named entities are annotated using the IOB (Inside, Outside, Beginning) tagging scheme.

- **Approach:**

  **Part 1: Character-Level Text Generation with RNNs**  
  - **Data Preprocessing**: The text is read, converted to lowercase, and unnecessary whitespace is removed. A mapping between characters and integer indices is created for character-level tokenization.
  - **Input-Target Sequence Generation**: The text is broken into input sequences of fixed length (`SEQ_LEN`), and the corresponding target characters are the next character in the sequence.
  - **RNN Model Design**: The model is built using recurrent layers such as LSTM or GRU. The output layer produces a probability distribution over the possible next characters. Temperature is applied to adjust randomness in text generation.
  - **Text Generation**: A callback is implemented to generate text at the end of each training epoch, varying the temperature to explore how the model's predictions evolve as the training progresses.
  - **Perplexity Calculation**: The perplexity of the model is calculated to evaluate how well the model predicts sequences, with lower perplexity indicating better performance.
  - **Results**: The model generates text in the style of Edward Lear, and various temperatures are tested to adjust randomness in the predictions.

  **Part 2: Named Entity Recognition (NER) with RNNs**  
  - **Data Preprocessing**: Sentences from the NER dataset are grouped, and each word is mapped to an integer index. Similarly, entity tags are mapped to integer indices. Padding is applied to ensure consistent sequence lengths for training.
  - **RNN Model Design**: The model includes an Embedding layer, a Bidirectional GRU layer for sequence modeling, and a Dense layer with softmax activation to predict entity tags for each word in the sentence.
  - **Training**: The model is trained using sequences of words and their corresponding entity tags. Early stopping is applied to prevent overfitting.
  - **Evaluation and Visualization**: F1 scores are calculated for each entity tag to assess the model's performance. Additionally, a Principal Component Analysis (PCA) is applied to visualize the learned representations of the bidirectional GRU layer.
  - **Results**: The model's predictions for entity tags are visualized, and its performance is evaluated across different categories such as geographical entities (B-geo), persons (B-per), and organizations (B-org).

- **Techniques Used:**
  - **Recurrent Neural Networks (RNNs)**: RNNs and their variants (LSTM, GRU) are used to model sequences of text and sentences. Bidirectional RNNs are employed to capture both forward and backward context for NER.
  - **Text Generation with Temperature**: The concept of temperature is applied to control the randomness in text generation, allowing for experimentation with more or less creative outputs.
  - **Named Entity Recognition**: NER is treated as a sequence labeling problem, where the model predicts tags for each word in a sentence.
  - **Visualization**: PCA is applied to visualize how the model represents different entity tags in its latent space.



### Assignment 6: Transformers (cs109b_hw6.ipynb)

- **Problem Statement:**
  In this assignment, we explore pre-trained transformers for text classification, focusing on a real-world systematic review task. The goal is to classify medical abstracts as either "irrelevant" or "not irrelevant" to help doctors sift through research papers on sexually transmitted infections (STIs) in women with HIV in sub-Saharan Africa. 

- **Dataset:**
  The dataset consists of three files:
  1. `review_78678_irrelevant.csv` - Contains abstracts classified as irrelevant.
  2. `review_78678_not_irrelevant_included.csv` - Abstracts classified as not irrelevant and included in the review.
  3. `review_78678_not_irrelevant_excluded.csv` - Abstracts classified as not irrelevant but excluded from further analysis.
  
  The dataset is combined and processed into training and validation sets for classification.

- **Approach:**

  **1. Preprocessing and Data Pipeline**
  
  - **Loading the Abstract Data:** We load the CSV files containing the abstract data, label them based on whether they are relevant or not (`0` for irrelevant, `1` for not irrelevant).
  
  - **Data Preprocessing:** The abstract data is cleaned, concatenated, and non-null rows are kept. Then, the data is split into training (90%) and validation (10%) sets, stratified by the target variable.
  
  - **Tokenization:** The BERT tokenizer is used to preprocess the text data. Each abstract is tokenized into input IDs and attention masks, with a maximum sequence length of 128 tokens. The processed inputs (`train_x_processed` and `validate_x_processed`) are passed into the model.
  
  - **Dataset Pipeline:** TensorFlow Dataset pipelines are created for both training and validation sets. The data is shuffled, batched, and prefetched to ensure efficient model training.

  **2. BERT Model**
  
  - **Model Architecture:** We load the pre-trained `bert-base-uncased` model from the Hugging Face library using the `TFAutoModelForSequenceClassification` API. This model is fine-tuned for binary classification, with a softmax activation on the output layer for classifying abstracts as relevant or irrelevant.
  
  - **Training:** The BERT model is trained using the Adam optimizer with a small learning rate (`5e-5`) for 5 epochs. Early stopping and model checkpointing are used to monitor validation performance and avoid overfitting. The training history (accuracy and loss) is plotted, showing the model's performance over the epochs.

  **3. DeBERTa Model**
  
  - **Model Architecture:** We repeat the same process as for the BERT model but use the DeBERTa V3 base model (`'microsoft/deberta-v3-base'`). This model is also fine-tuned for binary classification using similar tokenization, dataset pipelines, and model training steps.
  
  - **Training and Results:** The DeBERTa model is trained with the same hyperparameters as the BERT model. The training history for both models is compared, and the results are visualized.

  **4. Evaluation and Analysis**
  
  - **Confusion Matrices:** Confusion matrices are generated for both the BERT and DeBERTa models, providing insights into the types of errors each model makes. This helps us understand their strengths and weaknesses.
  
  - **Abstract Predictions:** For both models, we examine the top 4 abstracts predicted as highly relevant and the top 4 predicted as irrelevant. This qualitative analysis helps assess the models' ability to correctly classify the abstracts and provides insights into their decision-making.

  **5. Model Comparison and Discussion**
  
  - **Model Performance:** The training and validation histories for both models show good performance, with accuracies above 90%. The DeBERTa model demonstrates slightly better stability in validation accuracy and loss, suggesting better generalization. Both models exhibit high precision but some variance in recall.
  
  - **Confusion Matrix Analysis:** The confusion matrices highlight differences in false positives and false negatives between the models. DeBERTa shows fewer false positives but more false negatives, making it more conservative in classifying abstracts as relevant. BERT, on the other hand, has a more balanced error profile but tends to classify more abstracts as relevant.
  
  - **Abstract Evaluation:** The models are qualitatively evaluated based on their top predictions. BERT tends to select more epidemiologically relevant abstracts, while DeBERTa focuses on broader healthcare topics. Both models perform well, but their classification of abstracts reveals different areas of focus.

  **6. Positional Encoding in BERT vs. DeBERTa**
  
  - **Disentangled Attention:** DeBERTa introduces disentangled attention, where content and positional embeddings are separated, allowing for more flexible handling of relative positions in text.
  
  - **Enhanced Mask Decoder:** DeBERTa incorporates absolute positions in the decoding layer, which allows it to better distinguish between similar words in different positions.

- **Techniques Used:**
  - **Transformers:** Pre-trained BERT and DeBERTa models are fine-tuned for binary classification.
  - **Tokenization:** Both models use their respective tokenizers (BERT and DeBERTa) to preprocess text.
  - **Early Stopping and Model Checkpoints:** Callbacks are used to prevent overfitting and save the best model during training.
  - **Confusion Matrix Analysis:** The confusion matrix is used to evaluate the types of errors each model makes.
  
- **Challenges:**
  - **Computational Limitations:** Training large models like BERT and DeBERTa required significant computational resources. Memory issues and server interruptions slowed progress.
  - **Model Performance:** It was challenging to break through a validation accuracy of 87.5%, and fine-tuning hyperparameters was time-consuming.

