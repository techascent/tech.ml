(ns tech.verify.ml.regression
  (:require [tech.ml-base :as ml]
            [tech.ml.loss :as loss]
            [tech.ml.train :as train]
            [tech.ml.dataset :as dataset]
            [clojure.test :refer :all]))


(defn datasets
  []
  (let [f (partial * 2)
        observe (fn []
                  (let [x (- (* 20 (rand)) 10)
                        y (f x)]
                    {:x x :y y}))
        train-dataset (->> (repeatedly observe)
                           (take 1000))
        test-dataset (for [x (range -9.9 10 0.1)] {:x x :y (f x)})]
    {:train-ds train-dataset
     :test-ds test-dataset}))


(defn basic-regression
  [{:keys [model-type accuracy]
    :or {accuracy 0.01} :as options}]
  (let [{train-dataset :train-ds
         test-dataset :test-ds} (datasets)
        test-labels (map :y test-dataset)
        model (ml/train options [:x] :y train-dataset)
        test-output (ml/predict model test-dataset)
        mse (loss/mse test-output test-labels)]
    (is (< mse (double accuracy)))))


(defn scaled-features
  [options]
  (let [{train-dataset :train-ds
         test-dataset :test-ds} (datasets)
        test-labels (map :y test-dataset)
        model (ml/train (merge {:range-map {::dataset/features [-1 1]}}
                               options)
                        [:x] :y
                        train-dataset)
        test-output (ml/predict model test-dataset)
        mse (loss/mse test-output test-labels)]
    (is (< mse 0.01))))



(defn k-fold-regression
  [options]
  (let [{train-dataset :train-ds
         test-dataset :test-ds} (datasets)
        feature-keys [:x]
        label :y
        train-fn (partial ml/train options feature-keys label)
        predict-fn ml/predict
        mse (->> train-dataset
                 (dataset/->k-fold-datasets 10 {})
                 (train/average-prediction-error train-fn predict-fn
                                                 label loss/mse)
                 :average-loss)]
    (is (< mse 0.01))))


(defn auto-gridsearch-simple
  [options]
  ;;Pre-scale the dataset.
  (let [gs-options (ml/auto-gridsearch-options options)
        retval (ml/gridsearch [gs-options]
                              [:x] :y
                              loss/mse (:train-ds (datasets))
                              :scalar-labels? true
                              :gridsearch-depth (or (get options :gridsearch-depth)
                                                    100)
                              :range-map {::dataset/features [-1 1]})]
    (is (< (double (:average-loss (first retval)))
           (double (or (:mse-loss options)
                       0.2))))
    retval))
