(ns tech.verify.ml.regression
  (:require [tech.ml :as ml]
            [tech.ml.loss :as loss]
            [tech.ml.dataset :as dataset]
            [tech.ml.dataset.pipeline :as dsp]
            [tech.ml.dataset.pipeline.pipeline-operators
             :refer [without-recording
                     pipeline-train-context
                     pipeline-inference-context]]
            [clojure.test :refer :all]))


(defn mini-pipeline
  [dataset]
  (-> dataset
      dataset/->dataset
      (dsp/range-scale :x)
      (dataset/set-inference-target :y)))


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
         train-context :context} (pipeline-train-context
                                  (mini-pipeline train-dataset))
        {test-dataset :dataset} (pipeline-inference-context
                                 train-context
                                 (mini-pipeline test-dataset))]
    {:train-ds train-dataset
     :test-ds test-dataset}))



(defn basic-regression
  [{:keys [model-type accuracy]
    :or {accuracy 0.01} :as options}]
  (let [{train-dataset :train-ds
         test-dataset :test-ds} (datasets)
        model (ml/train options train-dataset)
        test-output (ml/predict model test-dataset)
        mse (loss/mse test-output (dataset/labels test-dataset))]
    (is (< mse (double accuracy)))))



(defn k-fold-regression
  [options]
  (let [{train-dataset :train-ds
         test-dataset :test-ds} (datasets)
        ave-result (->> (dataset/->k-fold-datasets train-dataset 10 options)
                        (ml/average-prediction-error options loss/mse))]
    (is (< (double (:average-loss ave-result)) 0.02))))


(defn auto-gridsearch-simple
  [options]
  ;;Pre-scale the dataset.
  (let [{train-dataset :train-ds
         test-dataset :test-ds} (datasets)
        gs-options (ml/auto-gridsearch-options options)
        retval (ml/gridsearch gs-options
                              loss/mse
                              train-dataset)]
    (is (< (double (:average-loss (first retval)))
           (double (or (:mse-loss options)
                       0.2))))
    retval))
