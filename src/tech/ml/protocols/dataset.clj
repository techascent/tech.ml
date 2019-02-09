(ns tech.ml.protocols.dataset
  (:require [clojure.set :as c-set]
            [tech.ml.protocols.column :as col-proto]
            [tech.datatype :as dtype]
            [clojure.core.matrix.protocols :as mp]))


(defprotocol PColumnarDataset
  (dataset-name [dataset])
  (maybe-column [dataset column-name]
    "Return either column if exists or nil.")
  (columns [dataset])
  (add-column [dataset column]
    "Error if columns exists")
  (remove-column [dataset col-name]
    "Failes quietly")
  (update-column [dataset col-name update-fn]
    "Update a column returning a new dataset.  update-fn is a column->column transformation.
Error if column does not exist.")
  (add-or-update-column [dataset column]
    "If column exists, replace.  Else append new column.")
  (select [dataset colname-seq index-seq]
    "Reorder/trim dataset according to this sequence of indexes.  Returns a new dataset.
colname-seq - either keyword :all or list of column names with no duplicates.
index-seq - either keyword :all or list of indexes.  May contain duplicates.")
  (index-value-seq [dataset]
    "Get a sequence of tuples:
[idx col-value-vec]

Values are in order of column-name-seq.  Duplicate names are allowed and result in
duplicate values.")
  (supported-stats [dataset]
    "Return the set of natively supported stats for the dataset.  This must be at least
#{:mean :variance :median :skew}.")
  (from-prototype [dataset table-name column-seq]
    "Create a new dataset that is the same type as this one but with a potentially
different table name and column sequence.  Take care that the columns are all of
the correct type."))


(defn column
  "Return the column or throw if it doesn't exist."
  [dataset column-name]
  (if-let [retval (maybe-column dataset column-name)]
    retval
    (throw (ex-info (format "Failed to find column: %s" column-name)
                    {:column-name column-name}))))


(defn select-columns
  [dataset col-name-seq]
  (select dataset col-name-seq :all))


(defn ds-filter
  [dataset predicate & [column-name-seq]]
  ;;interleave, partition count would also work.
  (->> (index-value-seq (select dataset (or column-name-seq :all) :all))
       (filter (fn [[idx col-values]]
                 (apply predicate col-values)))
       (map first)
       (select dataset :all)))


(defn ds-group-by
  [dataset key-fn & [column-name-seq]]
  (->> (index-value-seq (select dataset (or column-name-seq :all) :all))
       (group-by (fn [[idx col-values]]
                   (apply key-fn col-values)))
       (map first)
       (select dataset :all)))


(defn ds-map
  [dataset map-fn & [column-name-seq]]
  (->> (index-value-seq (select dataset (or column-name-seq :all) :all))
       (map (fn [[idx col-values]]
              (apply map-fn col-values)))))


(defn ->flyweight
  "Convert dataset to flyweight dataset.
  Flag indicates "
  [dataset & {:keys [column-name-seq
                     error-on-missing-values?]
              :or {column-name-seq :all
                   error-on-missing-values? true}}]
  (let [dataset (select dataset column-name-seq :all)
        column-name-seq (map col-proto/column-name (columns dataset))]
    (if error-on-missing-values?
      (ds-map dataset (fn [& args]
                        (zipmap column-name-seq args)))
      ;;Much slower algorithm
      (if-let [ds-columns (seq (columns dataset))]
        (let [ecount (long (apply min (map dtype/ecount ds-columns)))
              columns (columns dataset)]
          (for [idx (range ecount)]
            (->> (for [col columns]
                   [(col-proto/column-name col)
                    (when-not (col-proto/is-missing? col idx)
                      (col-proto/get-column-value col idx))])
                 (remove nil?)
                 (into {}))))))))


(defn ->row-major
  "Given a dataset and a map if desired key names to sequences of columns,
  produce a sequence of maps where each key name points to contiguous vector
  composed of the column values concatenated."
  [dataset key-colname-seq-map {:keys [datatype]
                                :or {datatype :float64}}]
  (let [key-val-seq (seq key-colname-seq-map)
        all-col-names (mapcat second key-val-seq)
        item-col-count-map (->> key-val-seq
                                (map (fn [[item-k item-col-seq]]
                                       (when (seq item-col-seq)
                                         [item-k (count item-col-seq)])))
                                (remove nil?)
                                vec)]
    (ds-map dataset
            (fn [& column-values]
              (->> item-col-count-map
                   (reduce (fn [[flyweight column-values] [item-key item-count]]
                             (let [contiguous-array (dtype/make-array-of-type datatype (take item-count column-values))]
                               (when-not (= (dtype/ecount contiguous-array)
                                            (long item-count))
                                 (throw (ex-info "Failed to get correct number of items" {:item-key item-key})))
                               [(assoc flyweight item-key contiguous-array)
                                (drop item-count column-values)]))
                           [{} column-values])
                   first))
            all-col-names)))


(defrecord GenericColumnarDataset [table-name columns]
  PColumnarDataset
  (dataset-name [dataset] table-name)
  (maybe-column [dataset column-name]
    (->> columns
         (filter #(= column-name (col-proto/column-name %)))
         first))

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

  (add-or-update-column [ctx column]
    (let [col-name (col-proto/column-name column)
          found-name (->> (map col-proto/column-name columns)
                          (filter #(= col-name %))
                          first)]
      (if found-name
        (update-column ctx col-name (constantly column))
        (add-column ctx column))))

  (select [dataset column-name-seq index-seq]
    (let [all-names (map col-proto/column-name columns)
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
                   (let [col (column dataset col-name)]
                     (if indexes
                       (col-proto/select col indexes)
                       col))))
            vec))))

  (index-value-seq [dataset]
    (let [col-value-seq (->> columns
                             (mapv (fn [col]
                                     (col-proto/column-values col))))]
      (->> (apply map vector col-value-seq)
           (map-indexed vector))))

  (supported-stats [dataset]
    (col-proto/supported-stats (first columns)))

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
