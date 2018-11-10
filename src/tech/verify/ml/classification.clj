(ns tech.verify.ml.classification
  (:require [clojure.string :as s]
            [clojure.java.io :as io]
            [camel-snake-kebab.core :refer [->kebab-case]]
            [tech.ml.dataset :as dataset]
            [tech.ml-base :as ml]
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


(def fruit-feature-keys [:color-score :height :mass :width])
(def fruit-label :fruit-name)


(defn classify-fruit
  [system-name options]
  (let [{:keys [train-ds test-ds]} (->> (fruit-dataset)
                                        (dataset/->train-test-split {}))
        model (ml/train system-name fruit-feature-keys fruit-label
                        (merge {:model-type :classification}
                               options)
                        train-ds)
        test-output (ml/predict model test-ds)
        labels (map fruit-label test-ds)]
    (is (< (or (:classification-accuracy
                options)
               0.7)
           (loss/classification-accuracy test-output labels)))))
