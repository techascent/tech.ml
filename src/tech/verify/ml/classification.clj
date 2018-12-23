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
  [options]
  (let [{:keys [train-ds test-ds]} (->> (fruit-dataset)
                                        (dataset/->train-test-split {}))
        model (ml/train (merge {:range-map {::dataset/features [-1 1]}}
                               options)
                        fruit-feature-keys fruit-label
                        train-ds)
        test-output (ml/predict model test-ds)
        labels (map fruit-label test-ds)]
    ;;Accuracy gets *better* as it increases.  This is the opposite of a loss!!
    (is (> (loss/classification-accuracy test-output labels)
           (or (:classification-accuracy
                options)
               0.7)))))


(defn auto-gridsearch-fruit
  [options]
  (let [gs-options (ml/auto-gridsearch-options options)
        retval (ml/gridsearch [gs-options]
                              fruit-feature-keys fruit-label
                              loss/classification-loss (fruit-dataset)
                              ;;Small k-fold because tiny dataset
                              :k-fold 3
                              :range-map {::dataset/features [-1 1]})]
    (is (< (double (:average-loss (first retval)))
           (double (or (:classification-loss options)
                       0.2))))
    retval))
