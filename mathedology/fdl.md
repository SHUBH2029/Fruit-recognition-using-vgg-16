# Methodology: Fruit Discrimination Loss (FDL)

## 1. Introduction

Traditional image classification models are commonly trained using Cross-Entropy Loss, which increases the probability assigned to the correct class. Although Cross-Entropy Loss is highly effective, it does not explicitly encourage a large confidence gap between the most probable prediction and competing classes.

To improve classification confidence and feature discrimination, a custom loss function named **Margin-Based Contrastive Cross-Entropy Loss** was developed and integrated into the Modified VGG16 architecture.

The proposed loss function combines three components:

1. Cross-Entropy Loss
2. Confidence Margin Penalty
3. Similarity-Aware Regularization

This hybrid objective encourages the model to make more confident predictions while simultaneously learning more discriminative feature representations.

---

## 2. Motivation

The standard Cross-Entropy Loss is given by

[
Loss = -\log(TrueProbability)
]

where:

* **TrueProbability** represents the predicted probability of the correct class.

Although Cross-Entropy Loss encourages correct predictions, it does not guarantee a significant difference between the highest predicted probability and the second-highest predicted probability.

For example:

### Prediction A

| Class         | Probability |
| ------------- | ----------- |
| Correct Class | 0.90        |
| Second Class  | 0.05        |

### Prediction B

| Class         | Probability |
| ------------- | ----------- |
| Correct Class | 0.51        |
| Second Class  | 0.49        |

Both predictions are correct, but Prediction A is much more confident than Prediction B.

To encourage stronger confidence and better feature separation, additional penalty terms were introduced.

---

## 3. Cross-Entropy Component

The first component of the proposed loss function is the standard Cross-Entropy Loss:

[
CrossEntropyLoss = -\log(TrueProbability)
]

This component ensures that the probability assigned to the correct class increases during training.

A lower value indicates better classification performance.

---

## 4. Confidence Margin Penalty

### Concept

A reliable classifier should maintain a sufficiently large gap between the highest predicted probability and the second-highest predicted probability.

Let:

* **HighestProbability** = largest predicted probability
* **SecondHighestProbability** = second largest predicted probability

The confidence gap is calculated as:

[
ConfidenceGap = HighestProbability - SecondHighestProbability
]

A larger confidence gap indicates a more confident prediction.

### Margin Penalty

A predefined margin value called **MarginThreshold** is introduced.

The margin penalty is computed as:

[
MarginPenalty =
\max(0,
MarginThreshold - ConfidenceGap)
]

### Behavior

If

[
ConfidenceGap \ge MarginThreshold
]

then the penalty becomes zero.

If

[
ConfidenceGap < MarginThreshold
]

then a positive penalty is added to the loss.

This mechanism encourages the model to produce predictions with a larger confidence gap.

---

## 5. Similarity-Aware Regularization

### Concept

Fruit classes often share similar visual characteristics such as color, texture, shape, and surface patterns.

As a result, their feature representations may overlap in the learned feature space.

To encourage better feature separation, a similarity-aware regularization term is introduced.

Let:

* **TrueClassFeature** represent the feature representation of the correct class.
* **CompetingClassFeature** represent the feature representation of a competing class.

The Euclidean distance between these feature representations is calculated as:

[
FeatureDistance =
\left|
TrueClassFeature -
CompetingClassFeature
\right|
]

### Similarity Penalty

The similarity-aware regularization term is defined as:

[
SimilarityPenalty =
e^{-FeatureDistance}
]

### Interpretation

When the feature representations are very close:

[
FeatureDistance \rightarrow 0
]

the similarity penalty approaches 1, increasing the total loss.

When the feature representations are far apart:

[
FeatureDistance
]

becomes large and the similarity penalty approaches 0.

This encourages the model to learn more distinct and separable feature representations for different fruit classes.

---

## 6. Final Proposed Loss Function

The complete Margin-Based Contrastive Cross-Entropy Loss is defined as:

[
TotalLoss =
-\log(TrueProbability)
+
MarginWeight
\times
\max(0,
MarginThreshold - ConfidenceGap)
+
SimilarityWeight
\times
e^{-FeatureDistance}
]

where:

| Parameter        | Description                                                 |
| ---------------- | ----------------------------------------------------------- |
| TrueProbability  | Probability assigned to the correct class                   |
| ConfidenceGap    | Difference between highest and second-highest probabilities |
| MarginThreshold  | Desired minimum confidence gap                              |
| FeatureDistance  | Distance between class feature representations              |
| MarginWeight     | Weight assigned to the margin penalty                       |
| SimilarityWeight | Weight assigned to the similarity penalty                   |

---

## 7. Training Algorithm

### Step 1

Receive:

* Ground truth labels
* Predicted probabilities
* Feature representations

### Step 2

Compute Cross-Entropy Loss using the predicted probability of the correct class.

### Step 3

Find:

* Highest predicted probability
* Second-highest predicted probability

### Step 4

Calculate the confidence gap.

### Step 5

Compute the margin penalty using the predefined margin threshold.

### Step 6

Calculate the distance between feature representations.

### Step 7

Compute the similarity-aware regularization term.

### Step 8

Combine all loss components using their corresponding weights.

### Step 9

Average the loss over the batch.

### Step 10

Perform backpropagation and update model parameters.

---

## 8. Advantages of the Proposed Loss Function

### Improved Prediction Confidence

Encourages a larger separation between the most probable class and competing classes.

### Better Feature Discrimination

Promotes greater separation between feature representations of different fruit categories.

### Reduced Class Ambiguity

Helps distinguish visually similar fruits more effectively.

### Enhanced Generalization

Produces more robust classification boundaries and reduces overfitting.

### Stable Optimization

Provides additional guidance during training through confidence and feature-space constraints.

---

## 9. Parameters Used

| Parameter         | Value                     |
| ----------------- | ------------------------- |
| Margin Threshold  | Experimental Value        |
| Margin Weight     | Experimental Value        |
| Similarity Weight | Experimental Value        |
| Backbone Network  | Modified VGG16            |
| Dataset           | Fruit Recognition Dataset |

---

## 10. Conclusion

A custom Margin-Based Contrastive Cross-Entropy Loss was developed to improve fruit classification performance. The proposed loss extends conventional Cross-Entropy Loss by introducing a confidence-margin constraint and a similarity-aware regularization term. The margin component encourages confident predictions, while the similarity component promotes better feature separation. Together, these mechanisms improve classification robustness, enhance feature discrimination, and contribute to better overall recognition performance on the fruit image dataset.