(ns tech.verify.ml.regression
  (:require [tech.ml :as ml]
            [tech.ml.loss :as loss]
            [tech.ml.dataset :as dataset]
            [tech.ml.dataset.etl :as etl]
            [clojure.test :refer :all]))


(def pipeline '[[range-scaler (not target?)]])


(defn datasets
  []
  (let [f (partial * 2)
        observe (fn []
                  (let [x (- (* 20 (rand)) 10)
                        y (f x)]
                    {:x x :y y}))
        train-dataset (->> (repeatedly observe)
                           (take 1000))
        test-dataset (for [x (range -9.9 10 0.1)] {:x x :y (f x)})
        {train-dataset :dataset
         inference-pipeline :pipeline
         options :options} (etl/apply-pipeline train-dataset pipeline {:target :y})
        test-dataset (-> (etl/apply-pipeline test-dataset inference-pipeline
                                             (assoc options :recorded? true))
                         :dataset)]
    {:train-ds train-dataset
     :test-ds test-dataset
     :options options}))



(defn basic-regression
  [{:keys [model-type accuracy]
    :or {accuracy 0.01} :as options}]
  (let [{train-dataset :train-ds
         test-dataset :test-ds
         dataset-options :options} (datasets)
        options (merge dataset-options options)
        model (ml/train options train-dataset)
        test-output (ml/predict model test-dataset)
        mse (loss/mse test-output (dataset/labels test-dataset options))]
    (is (< mse (double accuracy)))))



(defn k-fold-regression
  [options]
  (let [{train-dataset :train-ds
         test-dataset :test-ds
         ds-options :options} (datasets)
        options (merge options ds-options)
        ave-result (->> (dataset/->k-fold-datasets train-dataset 10 options)
                        (ml/average-prediction-error options loss/mse))]
    (is (< (double (:average-loss ave-result)) 0.01))))


(defn auto-gridsearch-simple
  [options]
  ;;Pre-scale the dataset.
  (let [{train-dataset :train-ds
         test-dataset :test-ds
         dataset-options :options} (datasets)
        gs-options (ml/auto-gridsearch-options (merge dataset-options options))
        retval (ml/gridsearch gs-options
                              loss/mse
                              train-dataset)]
    (is (< (double (:average-loss (first retval)))
           (double (or (:mse-loss options)
                       0.2))))
    retval))
