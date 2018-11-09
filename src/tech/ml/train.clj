(ns tech.ml.train
  (:require [tech.ml.dataset :as dataset]
            [tech.parallel :as parallel]))


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
                                (map #(dataset/get-dataset-item % label-key)))]
                (loss-fn predictions labels))))
       (apply +)
       (* (/ 1.0 (count dataset-seq)))))


(defn- expand-parameter-sequence
  [base-options param-key param-seq-map]
  (let [val-seq (get param-seq-map param-key)
        param-seq-map (dissoc param-seq-map param-key)]
    (->> val-seq
         (mapcat (fn [seq-val]
                   (let [base-options (assoc base-options param-key seq-val)]
                     (if-let [next-key (first (keys param-seq-map))]
                       (lazy-seq (expand-parameter-sequence base-options next-key param-seq-map))
                       [base-options])))))))


(defn options-seq
  "Given base options map and a map of parameter keyword -> value sequence
produce a sequence of options maps that does a cartesian join across all of
the parameter sequences"
  [base-options parameter-sequence-map]
  (if-let [first-key (first (keys parameter-sequence-map))]
    (expand-parameter-sequence base-options first-key parameter-sequence-map)
    base-options))


(defn find-best-options
  "Given a sequence of options and a sequence of datasets (for k-fold),
run them and return the best options.
train-fn: (train-fn options dataset) -> model
predict-fn: (predict-fn options dataset) -> prediction-sequence
label-key: key to get labels from dataset.
loss-fn: (loss-fn label-sequence prediction-sequence)-> double
  Lowest number wins."
  [train-fn predict-fn label-key loss-fn {:keys [parallelism]
                                          :or {parallelism (.availableProcessors
                                                            (Runtime/getRuntime))}}
   option-seq dataset-seq]
  (->> option-seq
       (parallel/queued-pmap
        parallelism
        (fn [options]
          {:options options
           :error (average-prediction-error
                   (partial train-fn options)
                   (partial predict-fn options)
                   label-key
                   loss-fn
                   dataset-seq) }))
       (reduce (fn [best-map {:keys [options error] :as next-map}]
                 (if (or (not best-map)
                         (< (double error)
                            (double (:error best-map))))
                   {})))))
