(ns tech.verify.ml.train
  (:require [tech.ml-base :as ml]
            [tech.ml.loss :as loss]
            [tech.ml.train :as train]
            [tech.ml.dataset :as dataset]
            [clojure.test :refer :all]))


(defn basic-regression
  [system-name & {:keys [model-type accuracy]
                  :or {model-type :regression
                       accuracy 0.01}}]
  (let [f (partial * 2)
        observe (fn []
                  (let [x (- (* 20 (rand)) 10)
                        y (f x)]
                    {:x x :y y}))
        train-dataset (->> (repeatedly observe)
                           (take 1000))
        test-dataset (for [x (range -9.9 10 0.1)] {:x x})
        test-labels (map (comp f :x) test-dataset)
        model (ml/train system-name [:x] :y
                        {:model-type (or model-type :regression)} train-dataset)
        test-output (ml/predict model test-dataset)
        mse (loss/mse test-output test-labels)]
    (is (< mse (double accuracy)))))


(defn scaled-features
  [system-name]
  (let [f (partial * 2)
        observe (fn []
                  (let [x (- (* 20 (rand)) 10)
                        y (f x)]
                    {:x x :y y}))
        train-dataset (->> (repeatedly observe)
                           (take 1000))
        test-dataset (for [x (range -9.9 10 0.1)] {:x x})
        test-labels (map (comp f :x) test-dataset)
        model (ml/train system-name [:x] :y {:model-type :regression
                                             :range-map [-1 1]} train-dataset)
        test-output (ml/predict model test-dataset)
        mse (loss/mse test-output test-labels)]
    (is (< mse 0.01))))



(defn k-fold-regression
  [system-name]
  (let [f (partial * 2)
        observe (fn []
                  (let [x (- (* 20 (rand)) 10)
                        y (f x)]
                    {:x x :y y}))
        dataset (->> (repeatedly observe)
                     (take 1000))
        feature-keys [:x]
        label :y
        train-fn (partial ml/train system-name feature-keys label
                          {:model-type :regression})
        predict-fn ml/predict
        mse (->> dataset
                 (dataset/dataset->k-fold-datasets 10 {})
                 (train/average-prediction-error train-fn predict-fn
                                                 label loss/mse))]
    (is (< mse 0.01))))


(defn gridsearch
  [system-name options]
  (let [f (partial * 2)
        observe (fn []
                  (let [x (- (* 20 (rand)) 10)
                        y (f x)]
                    {:x x :y y}))
        dataset (->> (repeatedly observe)
                     (take 1000))
        feature-keys [:x]
        label :y
        train-fn (partial ml/train system-name feature-keys label)
        predict-fn ml/predict
        k-fold-ds (dataset/dataset->k-fold-datasets 5 {} dataset)
        option-seq [(merge {:model-type :regression} options)
                    (merge {:model-type :regression} options)]
        {:keys [error options]} (train/find-best-options train-fn predict-fn
                                                         label
                                                         loss/mse {}
                                                         option-seq k-fold-ds)
        mse (or (:mse options) 0.01)]
    (is (< error mse))))
