(ns tech.verify.ml.classification
  (:require [clojure.string :as s]
            [clojure.java.io :as io]
            [camel-snake-kebab.core :refer [->kebab-case]]
            [tech.ml.dataset :as dataset]
            [tech.ml :as ml]
            [tech.ml.dataset.pipeline :as dsp]
            [tech.ml.dataset.pipeline.column-filters :as cf]
            [tech.ml.loss :as loss]
            [clojure.test :refer :all]))


(def fruit-dataset
  (memoize
   (fn []
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
                        (zipmap ds-keys))))
            dataset/->dataset)))))


(defn fruit-pipeline
  []
  (-> (fruit-dataset)
      (dataset/remove-columns [:fruit-subtype :fruit-label])
      (dsp/string->number)
      (dsp/range-scale #(cf/not % (cf/categorical? %)))
      (dataset/set-inference-target :fruit-name)))


(defn classify-fruit
  [options]
  (let [options (assoc options :target :fruit-name)
        dataset (fruit-pipeline)
        {:keys [train-ds test-ds]} (dataset/->train-test-split dataset {})
        model (ml/train options train-ds)
        test-output (ml/predict model test-ds)
        labels (dataset/labels test-ds)]

    ;;Accuracy gets *better* as it increases.  This is the opposite of a loss!!
    (is (> (loss/classification-accuracy test-output labels)
           (or (:classification-accuracy options)
               0.7)))))


(defn auto-gridsearch-fruit
  [options]
  (let [options (assoc options
                       :target :fruit-name
                       :k-fold 3)
        dataset (fruit-pipeline)
        ;; Annotate options with gridsearch information.
        gs-options (ml/auto-gridsearch-options options)
        retval (ml/gridsearch (assoc gs-options :k-fold 3)
                              loss/classification-loss
                              dataset)]
    (is (< (double (:average-loss (first retval)))
           (double (or (:classification-loss options)
                       0.2))))
    retval))
