(ns tech.ml.train)


(defn dataset-seq->dataset-model-seq
  "Given a sequence of {:train-ds ...} datasets, produce a sequence of:
  {:model ...}
  train-ds is removed to keep memory usage as low as possible.
  See dataset/dataset->k-fold-datasets"
  [train-fn dataset-seq]
  (->> dataset-seq
       (map (fn [{:keys [train-ds] :as item}]
              (-> (dissoc item :train-ds)
                  (assoc :model (train-fn train-ds)))))))


(defn average-prediction-error
  "Average prediction error across models generated with these datasets
  Page 242, https://web.stanford.edu/~hastie/ElemStatLearn/"
  [train-fn predict-fn label-key loss-fn dataset-seq]
  (->> (dataset-seq->dataset-model-seq train-fn dataset-seq)
       (map (fn [{:keys [test-ds model]}]
              (let [predictions (predict-fn model test-ds)
                    labels (->> test-ds
                                (map #(get-dataset-item % label-key)))]
                (loss-fn predictions labels))))
       (apply +)
       (* (/ 1.0 (count dataset-seq)))))
