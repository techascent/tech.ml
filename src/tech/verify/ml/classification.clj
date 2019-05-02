(ns tech.verify.ml.classification
  (:require [clojure.string :as s]
            [clojure.java.io :as io]
            [camel-snake-kebab.core :refer [->kebab-case]]
            [tech.ml.dataset :as ds]
            [tech.ml :as ml]
            [tech.ml.dataset.pipeline :as dsp]
            [tech.ml.dataset.pipeline.column-filters :as cf]
            [tech.ml.loss :as loss]
            [clojure.test :refer :all]))


(defn mapseq-dataset
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


(def fruit-dataset
  (memoize
   (fn []
     (ds/->dataset (mapseq-dataset)))))


(defn fruit-pipeline
  [dataset training?]
  (-> dataset
      (ds/remove-columns [:fruit-subtype :fruit-label])
      (dsp/range-scale #(cf/not cf/categorical?))
      (dsp/pwhen
       training?
       #(dsp/without-recording
         (-> %
             (dsp/string->number :fruit-name)
             (ds/set-inference-target :fruit-name))))))


(defn classify-fruit
  [options]
  (let [options (assoc options :target :fruit-name)
        pipeline-data (dsp/pipeline-train-context
                       (fruit-pipeline (fruit-dataset) true))
        ds (:dataset pipeline-data)
        {:keys [train-ds test-ds]} (ds/->train-test-split ds {})
        model (ml/train options train-ds)
        test-output (ml/predict model test-ds)
        labels (ds/labels test-ds)]

    ;;Accuracy gets *better* as it increases.  This is the opposite of a loss!!
    (is (> (loss/classification-accuracy test-output labels)
           (or (:classification-accuracy options)
               0.7)))
    ;;Now here is the production pathway
    (let [inference-src-ds (ds/remove-columns
                            (fruit-dataset)
                            [:fruit-name :fruit-subtype :fruit-label])
          inference-ds (-> (dsp/pipeline-inference-context
                            (:context pipeline-data)
                            (fruit-pipeline inference-src-ds false))
                           :dataset)
          inference-output (ml/predict model inference-ds)]
      (is (> (loss/classification-accuracy test-output labels)
             (or (:classification-accuracy options)
                 0.7))))))


(defn auto-gridsearch-fruit
  [options]
  (let [options (assoc options
                       :target :fruit-name
                       :k-fold 3)
        ds (fruit-pipeline (fruit-dataset) true)
        ;; Annotate options with gridsearch information.
        gs-options (ml/auto-gridsearch-options options)
        retval (ml/gridsearch (assoc gs-options :k-fold 3)
                              loss/classification-loss
                              ds)]
    (is (< (double (:average-loss (first retval)))
           (double (or (:classification-loss options)
                       0.2))))
    retval))
