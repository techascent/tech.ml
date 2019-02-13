(ns tech.ml.example.svm-datasets
  "These are datasets from this page:
  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"
  (:require [clojure.string :as s]
            [tech.ml.dataset.etl :as etl]
            [tech.ml.dataset.svm :as svm]
            [tech.ml.dataset :as ds]
            [tech.ml.dataset.column :as ds-col]))


(def basic-svm-pipeline '[[string->number string?]
                          [replace-missing * 0]
                          ;;scale everything [-1 1]
                          [range-scaler [not categorical?]]])

(def profile-pipeline '[[range-scaler [not categorical?]]])


(defn profile-range-scaler
  [dataset]
  (dotimes [iter 100]
    (time
     (etl/apply-pipeline dataset profile-pipeline {:target :label})))
  :ok)


(defn parse-svm-files
  [train-fname label-map & [test-fname]]
  (println "Loading" train-fname)
  (time
   (let [{train-ds :dataset
          options :options
          pipeline :pipeline}
         (-> (svm/parse-svm-file train-fname :label-map label-map)
             :dataset
             (etl/apply-pipeline basic-svm-pipeline {:target :label}))
         test-ds (when test-fname
                   (-> (svm/parse-svm-file test-fname :label-map label-map)
                       :dataset
                       (etl/apply-pipeline pipeline {:target :label
                                                     :recorded? true})
                       :dataset))]
     (merge {:train-ds train-ds
             :options options}
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
