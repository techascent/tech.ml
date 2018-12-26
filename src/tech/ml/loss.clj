(ns tech.ml.loss
  (:require [clojure.core.matrix :as m]))


(defn mse
  "mean squared error"
  ^double [predictions labels]
  (assert (= (count predictions) (count labels)))
  (let [n (count predictions)]
    (double
     (/ (-> (m/sub predictions labels)
            (m/pow 2)
            (m/esum))
        n))))


(defn rmse
  "root mean squared error"
  ^double [predictions labels]
  (-> (mse predictions labels)
      (Math/sqrt)))


(defn classification-accuracy
  "correct/total.
Model output is a sequence of probability distributions.
label-seq is a sequence of values.  The answer is considered correct
if the key highest probability in the model output entry matches
that label."
  ^double [model-output label-seq]
  (let [num-items (count model-output)
        num-correct (->> model-output
                         (map #(apply max-key % (keys %)))
                         (map vector label-seq)
                         (filter #(apply = %))
                         count)]
    (/ (double num-correct)
       (double num-items))))


(defn classification-loss
  "1.0 - classification-accuracy."
  ^double [model-output label-seq]
  (- 1.0
     (classification-accuracy model-output label-seq)))
