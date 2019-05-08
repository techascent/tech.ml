(ns tech.ml.regression
  "Utilities for training/verifying regression models"
  (:require [tech.ml :as ml]
            [tech.ml.loss :as loss]
            [tech.ml.model]
            [tech.ml.dataset :as dataset]
            [tech.v2.datatype.functional :as dfn]))


(def libsvm-regression-models
  (future (try
            (require '[tech.libs.svm])
            [:libsvm/regression]
            (catch Throwable e
              (println e)
              []))))


(def smile-regression-models
  (future (try
            (require '[tech.libs.smile.regression])
            [:smile.regression/ridge
             :smile.regression/lasso]
            (catch Throwable e []))))


(def xgboost-regression-models
  (future (try
            (require '[tech.libs.xgboost])
            [:xgboost/regression]
            (catch Throwable e []))))


(defn default-gridsearch-models
  []
  (->> (concat @libsvm-regression-models
               @smile-regression-models
               @xgboost-regression-models)))


(defn verify-model
  [trained-model test-ds loss-fn]
  (let [predictions (ml/predict trained-model test-ds)
        labels (dataset/labels test-ds)
        loss-val (loss-fn predictions labels)
        residuals (dfn/- labels predictions)]
        (merge
     {:loss loss-val
      :residuals (vec residuals)
      :predictions (vec predictions)
      :average-loss loss-val
      :labels labels}
     trained-model)))

(defn- ->option-map
  [model-options]
  (if (keyword? model-options)
    {:model-type model-options}
    model-options))


(defn train-regressors
  "Train a range of regressors across a dataset producing
  residuals and a set of information for each model."
  [dataset options
   & {:keys  [regression-systems
              gridsearch-regression-systems
              dataset-name
              loss-fn]
      :or {
           loss-fn loss/rmse
           dataset-name (dataset/dataset-name dataset)}}]
  (let [gridsearch-regression-systems (or gridsearch-regression-systems
                                          (default-gridsearch-models))
        train-test-split (dataset/->train-test-split dataset options)
        trained-results
        (concat (->> regression-systems
                     (map ->option-map)
                     (mapv (fn [model-options]
                             (println (format "Training dataset %s model %s"
                                              dataset-name (:model-type model-options)))
                             (let [best-model (ml/train (merge options model-options)
                                                        (:train-ds train-test-split))]
                               (verify-model best-model (:test-ds train-test-split) loss-fn)))))
                (->> gridsearch-regression-systems
                     (map ->option-map)
                     (mapv (fn [model-options]
                             (println (format "Gridsearching dataset %s model %s"
                                              dataset-name (:model-type model-options)))
                             (let [best-model (-> (ml/gridsearch (merge options
                                                                        model-options)
                                                                 loss-fn (:train-ds train-test-split))
                                                  first)]
                               (verify-model best-model (:test-ds train-test-split) loss-fn))))))]
    (vec trained-results)))
