(ns tech.ml.gridsearch
  "https://en.wikipedia.org/wiki/Sobol_sequence
  Used to gridsearch efficiently without getting fancy."
  (:require [tech.datatype :as dtype]
            [clojure.set :as c-set])
  (:import [org.apache.commons.math3.random SobolSequenceGenerator]))


(defn sobol-seq
  "Given a dimension count and optional start index, return an infinite sequence of
  points in the unit hypercube with coordinates in range [0-1].
  :start-index - starting index.  Constant time to start at any index.
  Returns sequence of double arrays of length n-dim."
  [n-dims gridsearch-start-index]
  (let [gen (SobolSequenceGenerator. n-dims)]
    (when gridsearch-start-index
      (.skipTo gen gridsearch-start-index))
    (repeatedly #(.nextVector gen))))


(defn make-gridsearch-fn
  [target-range grid-fn]
  (let [[tmin tmax] target-range
        tmin (double tmin)
        tmax (double tmax)
        trange (- tmax tmin)]
    ;;Value is in range 0-1
    (fn [^double value]
      (-> value
          (* trange)
          (+ tmin)
          grid-fn))))


(defn exp
  "Exponential exploration of the space."
  [item-range]
  (make-gridsearch-fn (mapv #(Math/log (double %)) item-range)
                      #(Math/exp (double %))))


(defn exp-long
  "Exponential exploration of the space."
  [item-range]
  (make-gridsearch-fn (mapv #(Math/log (double %)) item-range)
                      #(-> (double %)
                           Math/exp
                           Math/round
                           long)))


(defn linear
  "Linear search through the area."
  [item-range]
  (make-gridsearch-fn item-range identity))


(defn linear-long
  "Linear search through the area."
  [item-range]
  (make-gridsearch-fn item-range long))


(defn nominative
  "Non-numeric data.  Vector of options."
  [label-vec]
  (let [label-count (count label-vec)
        max-idx (dec label-count)]
    (make-gridsearch-fn [0 (max 0 (count label-vec))]
                        (fn [^double item-val]
                          (let [idx (min max-idx (long (Math/floor item-val)))]
                            (get label-vec idx))))))


(defn- map->path-value-seq*
  [data-map cur-path item-key]
  (let [data-item (get data-map item-key)
        cur-path (conj cur-path item-key)]
    (if (map? data-item)
      (->> (keys data-item)
           (mapcat (partial map->path-value-seq* data-item cur-path)))
      [[cur-path data-item]])))


(defn map->path-value-seq
  "Given a map, return a path-value seq where each path is the sequence
  of keys to get the value."
  [data-map]
  (->> (keys data-map)
       (mapcat (partial map->path-value-seq* data-map []))))


(defn path-item-seq->map
  [path-item-seq]
  (->> path-item-seq
       (reduce #(assoc-in %1 (first %2) (second %2)) {})))


(defn gridsearch
  "Given an option map return an infinite sequence of maps.  Values in the map
  that are fn? are considered valid options for the gridsearch."
  [option-map & [gridsearch-start-index]]
  (let [path-val-seq (map->path-value-seq option-map)
        constant-values (remove (comp fn? second) path-val-seq)
        dynamic-values (filterv (comp fn? second) path-val-seq)
        num-dynamic-values (count dynamic-values)]
    (if (= 0 num-dynamic-values)
      [(path-item-seq->map constant-values)]
      (->> (sobol-seq num-dynamic-values gridsearch-start-index)
           (map (fn [^doubles sobol-data]
                  (->> (concat constant-values
                               (->> dynamic-values
                                    (map-indexed
                                     (fn [idx [item-key item-fn]]
                                       [item-key (item-fn (aget sobol-data
                                                                (int idx)))]))))
                       path-item-seq->map)))))))
