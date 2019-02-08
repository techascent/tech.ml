(ns tech.ml.protocols.dataset
  (:require [clojure.set :as c-set]
            [tech.ml.protocols.column :as col-proto]))


(defprotocol PColumnarDataset
  (dataset-name [dataset])
  (column [dataset col-name])
  (columns [dataset])
  (add-column [dataset column])
  (remove-column [dataset col-name])
  (update-column [dataset col-name update-fn]
    "Update a column returning a new dataset.  update-fn is a column->column transformation.")
  (select [dataset index-seq]
    "Reorder/trim dataset according to this sequence of indexes.  Returns a new dataset.")
  (index-value-seq [dataset column-name-seq]
    "Get a sequence of tuples:
[idx col-value-vec]
values are in order of column-name-seq.  Duplicate names are allowed.")
  (supported-stats [dataset]
    "Return the set of natively supported stats for the dataset.  This must be at least
#{:mean :variance :median :skew}.")
  (from-prototype [dataset table-name column-seq]
    "Create a new dataset that is the same type as this one but with a potentially
different table name and column sequence.  Take care that the columns are all of
the correct type."))


(defn select-columns
  [dataset col-name-seq]
  (->> col-name-seq
       (map (partial column dataset))))


(defn ds-filter
  [dataset predicate column-name-seq]
  ;;interleave, partition count would also work.
  (->> (index-value-seq dataset column-name-seq)
       (filter (fn [[idx col-values]]
                 (apply predicate col-values)))
       (map first)
       (select dataset)))


(defn ds-group-by
  [dataset key-fn column-name-seq]
  (->> (index-value-seq dataset column-name-seq)
       (group-by (fn [[idx col-values]]
                   (apply key-fn col-values)))
       (map first)
       (select dataset)))


(defn ds-map
  [dataset map-fn column-name-seq]
  (->> (index-value-seq dataset column-name-seq)
       (map (fn [[idx col-values]]
              (apply map-fn col-values)))))


(defrecord GenericColumnarDataset [table-name columns]
  PColumnarDataset
  (dataset-name [dataset] table-name)
  (column [dataset column-name]
    (if-let [retval
             (->> columns
                  (filter #(= column-name (col-proto/column-name %)))
                  first)]
      retval
      (throw (ex-info (format "Failed to find column: %s" column-name)
                      {:column-name column-name}))))

  (columns [dataset] columns)

  (add-column [dataset col]
    (let [existing-names (set (map col-proto/column-name columns))
          new-col-name (col-proto/column-name col)]
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
                            (remove #(= (col-proto/column-name %)
                                        col-name)))))

  (update-column [ctx col-name col-fn]
    (->GenericColumnarDataset
     table-name
     (->> columns
          ;;Mapv to force failures in this function.
          (mapv (fn [col]
                  (if (= col-name (col-proto/column-name col))
                    (if-let [new-col (col-fn col)]
                      (do
                        (when-not (satisfies? col-proto/PColumn new-col)
                          (throw (ex-info (format "Column returned does not satisfy column protocols %s."
                                                  (type new-col))
                                          {})))
                        new-col)
                      (throw (ex-info (format "No column returned from column function %s."
                                              col-fn) {})))
                    col))))))

  (select [dataset index-seq]
    (let [idx-ary (int-array index-seq)]
      (->GenericColumnarDataset
       table-name
       (->> columns
            (mapv (fn [col]
                    (col-proto/select col idx-ary)))))))

  (index-value-seq [dataset column-name-seq]
    (let [col-value-seq (->> (select-columns dataset column-name-seq)
                             (map col-proto/column-values))]
      (->> (apply map vector col-value-seq)
           (map-indexed vector))))

  (supported-stats [dataset]
    ;;The minimum required.
    #{:mean :median :min :max :skew})

  (from-prototype [dataset table-name column-seq]
    (->GenericColumnarDataset table-name column-seq)))
