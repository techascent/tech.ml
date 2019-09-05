(ns tech.ml.classification
  (:require [tech.ml.dataset :as ds]
            [tech.v2.datatype.pprint :as dtype-pp]))


(defn probability-distributions->labels
  [prob-dists]
  (->> prob-dists
       (map (fn [pred-map]
              (first (apply max-key second pred-map))))))


(defn safe-inc
    [item]
    (if item
      (inc item)
      1))

(defn confusion-map
  [predicted-labels labels]
  (let [answer-counts (frequencies labels)]
    (->> (map vector predicted-labels labels)
         (reduce (fn [total-map [pred actual]]
                   (update-in total-map [actual pred]
                              safe-inc))
                 {})
         (map (fn [[k v]]
                [k (->> v
                        (map (fn [[guess v]]
                               [guess
                                (double (/ v (get answer-counts k)))]))
                        (into (sorted-map)))]))
         (into (sorted-map)))))


(defn confusion-map->ds
  [conf-matrix-map]
  (let [all-labels (->> (keys conf-matrix-map)
                        sort)
        header-column (merge {:column-name "column-name"}
                             (->> all-labels
                                  (map #(vector % %))
                                  (into {})))
        column-names (concat [:column-name]
                             all-labels)]
    (->> all-labels
         (map (fn [label-name]
                (let [entry (get conf-matrix-map label-name)]
                  (merge {:column-name label-name}
                         (->> all-labels
                              (map (fn [entry-name]
                                     [entry-name (dtype-pp/format-object
                                                  (get entry entry-name 0.0))]))
                              (into {}))))))
         (concat [header-column])
         (ds/->>dataset)
         ;;Ensure order is consistent
         (#(ds/select-columns % column-names)))))
