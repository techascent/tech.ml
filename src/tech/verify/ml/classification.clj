(ns tech.verify.ml.classification
  (:require [clojure.string :as s]
            [clojure.java.io :as io]
            [camel-snake-kebab.core :refer [->kebab-case]]
            [tech.ml.dataset :as dataset]))


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
