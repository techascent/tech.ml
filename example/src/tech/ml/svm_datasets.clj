(ns tech.ml.svm-datasets
  "These are datasets from this page:
  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"
  (:require [clojure.string :as s]))


(def test-line "1 1:2.617300e+01 2:5.886700e+01 3:-1.894697e-01 4:1.251225e+02")


(defn- parse-line
  [line]
  (let [parse (s/split line #"\s+")
        label (Double/parseDouble (first parse))
        idx-val-pairs (->> (rest parse)
                           (map (fn [item]
                                  (let [[idx val] (s/split item #":")]
                                    [(Long/parseLong idx)
                                     (Double/parseDouble val)]))))]
    {:label label
     :features (->> idx-val-pairs
                    (into {}))
     :max-idx (apply max (map first idx-val-pairs))
     :min-idx (apply min (map first idx-val-pairs))}))


(defn parse-svm-file
  "Parse an svm file and generate a dataset of {:features :label}.  Always
  represents result as a dense matrix; no support for sparse datasets."
  [fname & [label-map]]
  (let [f-data (slurp fname)
        [labels features min-idx max-idx]
        (->> (s/split f-data #"\n")
             (pmap parse-line)
             (reduce
              (fn [[labels features min-idx max-idx] next-line]
                (let [{label :label
                       line-feature :features
                       line-max-idx :max-idx
                       line-min-idx :min-idx
                       :as item} next-line]
                  [(conj labels label)
                   (conj features line-feature)
                   (if (and min-idx
                            (< min-idx line-min-idx))
                     min-idx
                     line-min-idx)
                   (if (and max-idx
                            (> max-idx line-max-idx))
                     max-idx
                     line-max-idx)]))
              [[] [] nil nil]))
        min-idx (long min-idx)
        max-idx (long max-idx)
        num-items (+ 1 (- max-idx min-idx))]
    (map (fn [label features]
           (let [double-ary (double-array num-items)]
             (doseq [[idx val] features]
               (aset double-ary (- (long idx)
                                   min-idx)
                     (double val)))
             {:label (if label-map
                       (get label-map (double label))
                       (double label))
              :features double-ary}))
         labels features)))


(defn leukemia
  "Low N high feature space dataset"
  []
  (let [label-map {-1.0 :negative
                   1.0 :positive}]
    {:train-ds (parse-svm-file "data/leu" label-map)
     :test-ds (parse-svm-file "data/leu.t" label-map)
     :type :classification
     :name :leukemia}))


(defn duke-breast-cancer
  "Low N high feature space dataset"
  []
  (let [label-map {-1.0 :negative
                   1.0 :positive}]
    {:train-ds (parse-svm-file "data/duke.tr" label-map)
     :test-ds (parse-svm-file "data/duke.val" label-map)
     :type :classification
     :name :duke-breast-cancer}))


(defn test-ds-1
  []
  (let [label-map {0.0 :negative
                   1.0 :positive}]
    {:train-ds (parse-svm-file "data/train.1" label-map)
     :test-ds (parse-svm-file "data/test.1" label-map)
     :type :classification
     :name :test-dataset-1}))


(defn test-ds-2
  []
  (let [label-map {1.0 :first
                   2.0 :second
                   3.0 :third}]
    {:type :classification
     :train-ds (parse-svm-file "data/train.2" label-map)
     :name :test-dataset-2}))


(defn test-ds-3
  []
  (let [label-map {-1.0 :negative
                   1.0 :positive}]
    {:type :classification
     :train-ds (parse-svm-file "data/train.3" label-map)
     :test-ds (parse-svm-file "data/test.3" label-map)
     :name :test-dataset-3}))



(defn all-datasets
  []
  (->> [leukemia
        duke-breast-cancer
        test-ds-1
        test-ds-2
        test-ds-3]
       (map (fn [ds-fn]
              (ds-fn)))))
