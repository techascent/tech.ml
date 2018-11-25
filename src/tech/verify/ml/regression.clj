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
  [system-name & {:keys [model-type accuracy]
                  :or {model-type :regression
                       accuracy 0.01}}]
  (let [{train-dataset :train-ds
         test-dataset :test-ds} (datasets)
        test-labels (map :y test-dataset)
        model (ml/train system-name [:x] :y
                        {:model-type (or model-type :regression)} train-dataset)
        test-output (ml/predict model test-dataset)
        mse (loss/mse test-output test-labels)]
    (is (< mse (double accuracy)))))


(defn scaled-features
  [system-name]
  (let [{train-dataset :train-ds
         test-dataset :test-ds} (datasets)
        test-labels (map :y test-dataset)
        model (ml/train system-name [:x] :y {:model-type :regression
                                             :range-map {:dataset/features [-1 1]}}
                        train-dataset)
        test-output (ml/predict model test-dataset)
        mse (loss/mse test-output test-labels)]
    (is (< mse 0.01))))



(defn k-fold-regression
  [system-name]
  (let [{train-dataset :train-ds
         test-dataset :test-ds} (datasets)
        feature-keys [:x]
        label :y
        train-fn (partial ml/train system-name feature-keys label
                          {:model-type :regression})
        predict-fn ml/predict
        mse (->> train-dataset
                 (dataset/->k-fold-datasets 10 {})
                 (train/average-prediction-error train-fn predict-fn
                                                 label loss/mse))]
    (is (< mse 0.01))))


(defn auto-gridsearch-simple
  [system-name options]
  ;;Pre-scale the dataset.
  (let [gs-options (ml/auto-gridsearch-options
                    system-name
                    (merge {:model-type :regression}
                           options))
        retval (ml/gridsearch [[system-name gs-options]]
                              [:x] :y
                              loss/mse (:train-ds (datasets))
                              :scalar-labels? true
                              :gridsearch-depth (or (get options :gridsearch-depth)
                                                    100)
                              :range-map {::dataset/features [-1 1]})]
    (is (< (double (:error (first retval)))
           (double (or (:mse-loss options)
                       0.2))))
    retval))
