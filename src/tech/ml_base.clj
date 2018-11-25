(ns tech.ml-base
  (:require [tech.ml.registry :as registry]
            [tech.ml.protocols :as protocols]
            [tech.ml.dataset :as dataset]
            [tech.ml.gridsearch :as ml-gs]
            [tech.ml.train :as train]
            [tech.parallel :as parallel]
            [tech.datatype :as dtype]
            [clojure.set :as c-set])
  (:import [java.util UUID]))


(defn train
  [system-name feature-keys label-keys options dataset]
  (let [ml-system (registry/system system-name)
        options (merge options (protocols/coalesce-options ml-system options))
        {:keys [coalesced-dataset options]}
        (dataset/apply-dataset-options feature-keys label-keys options dataset)
        model (protocols/train ml-system options coalesced-dataset)]
    {:system system-name
     :model model
     :options options
     :feature-keys feature-keys
     :label-keys label-keys
     :id (UUID/randomUUID)}))


(defn predict
  [model dataset]
  (let [ml-system (registry/system (:system model))
        trained-model (:model model)
        {:keys [coalesced-dataset]} (dataset/apply-dataset-options
                                     (:feature-keys model) nil (:options model) dataset)]
    (protocols/predict ml-system
                       (:options model)
                       (:model model)
                       coalesced-dataset)))


(defn auto-gridsearch-options
  [system-name options]
  (let [ml-system (registry/system system-name)]
    (merge options
           (protocols/gridsearch-options ml-system options))))


(defn gridsearch
  "Gridsearch these system/option pairs by this dataset, averaging the errors
  across k-folds and taking the lowest top-n options.
We are breaking out of 'simple' and into 'easy' here, this is pretty
opinionated.  The point is to make 80% of the cases work great on the
first try."
  [system-name->options-seq feature-keys label-keys
   loss-fn dataset
   & {:keys [parallelism top-n gridsearch-depth k-fold
             scalar-labels?]
      :or {parallelism (.availableProcessors
                        (Runtime/getRuntime))
           top-n 5
           gridsearch-depth 50
           k-fold 5}
      :as options}]
  ;;Scale the dataset once; scanning it to find ranges of things is expensive.
  ;;You are free, however, to provide your own scale map in the options.
  (let [{:keys [options coalesced-dataset]}
        (dataset/apply-dataset-options feature-keys label-keys
                                       options
                                       dataset)
        ;;This makes mse work out later
        coalesced-dataset (if scalar-labels?
                            (->> coalesced-dataset
                                 (map (fn [ds-entry]
                                        (update ds-entry
                                                ::dataset/label #(dtype/get-value % 0)))))
                            coalesced-dataset)
        dataset-seq (if k-fold
                      (dataset/->k-fold-datasets k-fold options coalesced-dataset)
                      [coalesced-dataset])
        train-fn (fn [[system-name options-map] dataset]
                   (train system-name ::dataset/features ::dataset/label options-map dataset))
        predict-fn predict
        ;;Becase we are working with a
        ds-entry->predict-fn (if-let [label-map
                                      (get-in options [:label-map
                                                       (first (dataset/normalize-keys
                                                               label-keys))])]
                               ;;classification
                               (let [val->label (c-set/map-invert label-map)]
                                 (fn [{:keys [::dataset/label]}]
                                   (get val->label (-> (dtype/get-value label 0)
                                                       long))))
                               (do
                                 (fn [{:keys [::dataset/label]}]
                                   (dtype/get-value label 0))))]
    (->> system-name->options-seq
         ;;Build master set of gridsearch pairs
         (mapcat (fn [[system-name options-map]]
                   (->> (ml-gs/gridsearch options-map)
                        (take gridsearch-depth)
                        (map (fn [gs-opt] [system-name (merge options gs-opt)])))))
         (parallel/queued-pmap
          0
          (fn [sys-op-pair]
            {:system (first sys-op-pair)
             :options (second sys-op-pair)
             :error (train/average-prediction-error
                     (partial train-fn sys-op-pair)
                     predict-fn
                     ds-entry->predict-fn
                     loss-fn
                     dataset-seq)}))
         ;;Partition to keep sorting down a bit.
         (partition-all top-n)
         (reduce (fn [best-items next-group]
                   (->> (concat best-items next-group)
                        (sort-by :error)
                        (take top-n)))
                 []))))
