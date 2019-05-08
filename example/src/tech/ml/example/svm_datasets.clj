(ns tech.ml.example.svm-datasets
  "These are datasets from this page:
  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"
  (:require [clojure.string :as s]
            [tech.ml.dataset.svm :as svm]
            [tech.ml.dataset :as ds]
            [tech.ml.dataset.pipeline :as dsp]
            [tech.ml.dataset.pipeline.pipeline-operators
             :refer [without-recording
                     pipeline-train-context
                     pipeline-inference-context]]
            [tech.ml.dataset.column :as ds-col]
            [tech.v2.datatype :as dtype]))


(def basic-svm-pipeline '[[string->number string?]
                          [replace-missing * 0]
                          ;;scale everything [-1 1]
                          [range-scaler [not categorical?]]])


(defn profile-range-scaler
  [dataset]
  (dotimes [iter 100]
    (time (dsp/range-scale dataset)))
  :ok)

(defn svm-pipeline
  [dataset]
  (-> (ds/->dataset (:dataset dataset))
      (dsp/string->number)
      (dsp/replace-missing :all 0)
      (dsp/range-scale)
      (ds/set-inference-target :label)))


(defn parse-svm-files
  [train-fname label-map & [test-fname]]
  (println "Loading" train-fname)
  (time
   (let [{train-ds :dataset
          train-context :context}
         (pipeline-train-context
          (-> (svm/parse-svm-file train-fname :label-map label-map)
              svm-pipeline))
         {test-ds :dataset} (when test-fname
                              (pipeline-inference-context
                               train-context
                               (-> (svm/parse-svm-file test-fname :label-map label-map)
                                   svm-pipeline)))]
     (merge {:train-ds train-ds}
            (when test-ds {:test-ds test-ds})))))



(defn leukemia
  "Low N high feature space dataset"
  []
  (merge (parse-svm-files "data/leu" {-1.0 :negative
                                      1.0 :positive}
                          "data/leu.t")
         {:type :classification
          :name :leukemia}))


(defn duke-breast-cancer
  "Low N high feature space dataset"
  []
  (merge (parse-svm-files "data/duke.tr" {-1.0 :negative
                                          1.0 :positive}
                          "data/duke.val")
         {:type :classification
          :name :duke-breast-cancer}))


(defn test-ds-1
  []
  (merge (parse-svm-files "data/train.1" {0.0 :negative
                                          1.0 :positive}
                          "data/test.1")
         {:type :classification
          :name :test-dataset-1}))


(defn test-ds-2
  []
  (merge (parse-svm-files "data/train.2" {1.0 :first
                                          2.0 :second
                                          3.0 :third})
         {:type :classification
          :name :test-dataset-2}))


(defn test-ds-3
  []
  (merge (parse-svm-files "data/train.3" {-1.0 :negative
                                          1.0 :positive}
                          "data/test.3")
         {:type :classification
          :name :test-dataset-3}))



(def all-datasets
  (memoize
   (fn []
     (->> [leukemia
           duke-breast-cancer
           test-ds-1
           test-ds-2
           test-ds-3]
          (mapv (fn [ds-fn]
                 (ds-fn)))))))
