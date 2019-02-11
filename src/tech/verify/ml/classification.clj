(ns tech.verify.ml.classification
  (:require [clojure.string :as s]
            [clojure.java.io :as io]
            [camel-snake-kebab.core :refer [->kebab-case]]
            [tech.ml.dataset :as dataset]
            [tech.ml :as ml]
            [tech.ml.dataset.etl :as etl]
            [tech.ml.loss :as loss]
            [clojure.test :refer :all]))


(defn fruit-dataset
  []
  (let [fruit-ds (slurp (io/resource "fruit_data_with_colors.txt"))
        dataset (->> (s/split fruit-ds #"\n")
                     (mapv #(s/split % #"\s+")))
        ds-keys (->> (first dataset)
                     (mapv (comp keyword ->kebab-case)))]
    (->> (rest dataset)
         (map (fn [ds-line]
                (->> ds-line
                     (map (fn [ds-val]
                            (try
                              (Double/parseDouble ^String ds-val)
                              (catch Throwable e
                                (-> (->kebab-case ds-val)
                                    keyword)))))
                     (zipmap ds-keys)))))))


(def fruit-pipeline '[[remove :fruit-subtype]
                      [string->number string?]
                      ;;Range numeric data to -1 1
                      [range-scaler (not categorical?)]])


(defn classify-fruit
  [options]
  (let [options (assoc options :target :fruit-name)
        {:keys [dataset pipeline options]}
        (-> (fruit-dataset)
            (etl/apply-pipeline fruit-pipeline options))
        {:keys [train-ds test-ds]} (dataset/->train-test-split dataset {})
        model (ml/train options train-ds)
        test-output (ml/predict model test-ds)
        labels (dataset/labels test-ds options)]

    ;;Accuracy gets *better* as it increases.  This is the opposite of a loss!!
    (is (> (loss/classification-accuracy test-output labels)
           (or (:classification-accuracy options)
               0.7)))))


(defn auto-gridsearch-fruit
  [options]
  (let [options (assoc options
                       :target :fruit-name
                       :k-fold 3)
        {:keys [dataset pipeline options]}
        (-> (fruit-dataset)
            (etl/apply-pipeline fruit-pipeline options))
        ;; Annotate options with gridsearch information.
        gs-options (ml/auto-gridsearch-options options)
        retval (ml/gridsearch (assoc gs-options :k-fold 3)
                              loss/classification-loss
                              dataset)]
    (is (< (double (:average-loss (first retval)))
           (double (or (:classification-loss options)
                       0.2))))
    retval))
