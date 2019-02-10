(ns tech.ml.dataset.generic-columnar-dataset
  (:require [tech.ml.dataset.column :as ds-col]
            [tech.ml.dataset :as ds]
            [tech.ml.protocols.dataset :as ds-proto]
            [clojure.core.matrix.protocols :as mp]
            [clojure.set :as c-set]))


(defrecord GenericColumnarDataset [table-name columns]
  ds-proto/PColumnarDataset
  (dataset-name [dataset] table-name)
  (maybe-column [dataset column-name]
    (->> columns
         (filter #(= column-name (ds-col/column-name %)))
         first))

  (columns [dataset] columns)

  (add-column [dataset col]
    (let [existing-names (set (map ds-col/column-name columns))
          new-col-name (ds-col/column-name col)]
      (when-let [existing (existing-names new-col-name)]
        (throw (ex-info (format "Column of same name (%s) already exists in columns"
                                new-col-name)
                        {:existing-columns existing-names
                         :column-name new-col-name})))
      (->GenericColumnarDataset
       table-name
       (concat columns [col]))))

  (remove-column [dataset col-name]
    (->GenericColumnarDataset table-name
                       (->> columns
                            (remove #(= (ds-col/column-name %)
                                        col-name)))))

  (update-column [dataset col-name col-fn]
    (->GenericColumnarDataset
     table-name
     (->> columns
          ;;Mapv to force failures in this function.
          (mapv (fn [col]
                  (if (= col-name (ds-col/column-name col))
                    (if-let [new-col (col-fn col)]
                      (do
                        (when-not (ds-col/is-column? new-col)
                          (throw (ex-info (format "Column returned does not satisfy column protocols %s."
                                                  (type new-col))
                                          {})))
                        new-col)
                      (throw (ex-info (format "No column returned from column function %s."
                                              col-fn) {})))
                    col))))))

  (add-or-update-column [dataset column]
    (let [col-name (ds-col/column-name column)
          found-name (->> (map ds-col/column-name columns)
                          (filter #(= col-name %))
                          first)]
      (if found-name
        (ds/update-column dataset col-name (constantly column))
        (ds/add-column dataset column))))

  (select [dataset column-name-seq index-seq]
    (let [all-names (map ds-col/column-name columns)
          all-name-set (set all-names)
          column-name-seq (if (= :all column-name-seq)
                            all-names
                            column-name-seq)
          name-set (set column-name-seq)
          _ (when-let [missing (seq (c-set/difference name-set all-name-set))]
              (throw (ex-info (format "Invalid/missing column names: %s" missing)
                              {:all-columns all-name-set
                               :selection column-name-seq})))
          _ (when-not (= (count name-set)
                         (count column-name-seq))
              (throw (ex-info "Duplicate column names detected" {:selection column-name-seq})))
          indexes (if (= :all index-seq)
                    nil
                    (int-array index-seq))]
      (->GenericColumnarDataset
       table-name
       (->> column-name-seq
            (map (fn [col-name]
                   (let [col (ds/column dataset col-name)]
                     (if indexes
                       (ds-col/select col indexes)
                       col))))
            vec))))

  (index-value-seq [dataset]
    (let [col-value-seq (->> columns
                             (mapv (fn [col]
                                     (ds-col/column-values col))))]
      (->> (apply map vector col-value-seq)
           (map-indexed vector))))

  (supported-stats [dataset]
    (ds-col/supported-stats (first columns)))

  (from-prototype [dataset table-name column-seq]
    (->GenericColumnarDataset table-name column-seq))


  mp/PDimensionInfo
  (dimensionality [m] (count (mp/get-shape m)))
  (get-shape [m]
    [(if-let [first-col (first columns)]
       (mp/element-count first-col)
       0)
     (count columns)])
  (is-scalar? [m] false)
  (is-vector? [m] true)
  (dimension-count [m dimension-number]
    (let [shape (mp/get-shape m)]
      (if (<= (count shape) (long dimension-number))
        (get shape dimension-number)
        (throw (ex-info "Array does not have specific dimension"
                        {:dimension-number dimension-number
                         :shape shape}))))))
