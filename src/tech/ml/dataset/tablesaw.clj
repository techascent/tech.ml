(ns tech.ml.dataset.tablesaw
  (:require [tech.datatype.tablesaw :as dtype-tbl]
            [tech.ml.protocols.dataset :as ds-proto]
            [tech.ml.protocols.column :as col-proto]
            [tech.ml.protocols.etl :as etl-proto]
            [camel-snake-kebab.core :refer [->kebab-case]]
            [clojure.core.matrix.protocols :as mp]
            [tech.ml.dataset.etl :as etl]
            [tech.datatype :as dtype]
            [tech.datatype.base :as dtype-base]
            [tech.datatype.java-primitive :as primitive]
            [tech.parallel :as parallel]
            [clojure.set :as c-set]
            [tech.compute.tensor :as ct]
            [tech.compute.cpu.tensor-math :as cpu-tm]
            [tech.compute.cpu.typed-buffer :as cpu-typed-buffer]
            [tech.ml.dataset.etl.column-filters :as column-filters])
  (:import [tech.tablesaw.api Table ColumnType
            NumericColumn DoubleColumn
            StringColumn BooleanColumn]
           [tech.tablesaw.columns Column]
           [tech.tablesaw.io.csv CsvReadOptions]
           [tech.ml.protocols.dataset GenericColumnarDataset]
           [java.util UUID]
           [org.apache.commons.math3.stat.descriptive.moment Skewness])
  (:import [tech.compute.cpu UnaryOp BinaryOp]))



(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn- create-table-from-column-seq
  [^Table table col-seq]
  (Table/create (.name table) (into-array Column col-seq)))


(defn ->nice-name
  [item]
  (cond
    (keyword? item)
    (name item)
    (symbol? item)
    (name item)
    :else
    (str item)))


(defn- col-datatype-cast
  [data-val col-dtype]
  (if-let [dtype ((set primitive/datatypes) col-dtype)]
    (dtype/cast data-val dtype)
    (case col-dtype
      :string (if (or (keyword? data-val)
                      (symbol? data-val))
                (name data-val)
                (str data-val))
      :boolean (boolean data-val))))


(defn- column->metadata
  [^Column col]
  {:name (.name col)}
  (when (= :string (dtype/get-datatype col))
    {:categorical? true}))


(cpu-tm/add-unary-op! :log1p (reify UnaryOp
                               (op [this val]
                                 (Math/log1p (double val)))))

(cpu-tm/add-binary-op! :** (reify BinaryOp
                             (op [this lhs rhs]
                               (Math/pow lhs rhs))))


(defrecord ComputeTensorMathContext []
  col-proto/PColumnMathContext
  (unary-op [ctx op-env op-arg op-kwd]
    (ct/unary-op! (ct/clone op-arg) 1.0 op-arg op-kwd))

  (binary-op [ctx op-env op-args op-scalar-fn op-kwd]
    (let [first-pair (take 2 op-args)
          op-args (drop 2 op-args)
          [first-arg second-arg] first-pair
          any-tensors (->> op-args
                           (filter ct/acceptable-tensor-buffer?)
                           seq)
          accumulator (ct/from-prototype (first any-tensors))]
        (if (or (ct/acceptable-tensor-buffer? first-arg)
                (ct/acceptable-tensor-buffer? second-arg))
          (ct/binary-op! accumulator 1.0 first-arg 1.0 second-arg op-kwd)
          (ct/assign! accumulator (op-scalar-fn first-arg second-arg)))
        (reduce (fn [accumulator next-arg]
                  (ct/binary-op! accumulator 1.0 accumulator 1.0 next-arg op-kwd))
                accumulator
                op-args))))


(def available-stats
  (set [:mean
        :variance
        :median
        :min
        :max
        :skew
        :kurtosis
        :geometric-mean
        :sum-of-squares
        :sum-of-logs
        :quadratic-mean
        :standard-deviation
        :population-variance
        :sum
        :product
        :quartile-1
        :quartile-3]))


(defrecord TablesawColumn [^Column col metadata]
  col-proto/PIsColumn
  (is-column? [this] true)

  col-proto/PColumn
  (column-name [this] (.name col))
  (set-name [this colname]
    ;;Stupid way of doing it...but I don't have a better option at this point.
    (-> (col-proto/new-column this
                              (dtype/get-datatype this)
                              (col-proto/column-values this)
                              colname)))

  (supported-stats [col] available-stats)

  (metadata [this] (assoc metadata
                          :size (mp/element-count col)
                          :datatype (dtype/get-datatype col)
                          :name (.name col)))

  (set-metadata [this data-map]
    (->TablesawColumn col data-map))

  (missing [this]
    (-> (.isMissing ^Column col)
        (.toArray)))

  (unique [this]
    (-> (.unique ^Column col)
        (.asList)
        set))

  (stats [this stats-set]
    (when-not (instance? NumericColumn col)
      (throw (ex-info "Stats aren't available on non-numeric columns" {})))
    (let [stats-set (set (if-not (seq stats-set)
                           available-stats
                           stats-set))
          existing (->> stats-set
                        (map (fn [skey]
                               (when-let [cached (get metadata skey)]
                                 [skey cached])))
                        (remove nil?)
                        (into {}))
          missing-stats (c-set/difference stats-set (set (keys existing)))
          ^NumericColumn col col]
      (merge existing
             (->> missing-stats
                  (map (fn [skey]
                         [skey
                          (case skey
                            :mean (.mean col)
                            :variance (.variance col)
                            :median (.median col)
                            :min (.min col)
                            :max (.max col)
                            :skew (.skewness col)
                            :kurtosis (.kurtosis col)
                            :geometric-mean (.geometricMean col)
                            :sum-of-squares (.sumOfSquares col)
                            :sum-of-logs (.sumOfLogs col)
                            :quadratic-mean (.quadraticMean col)
                            :standard-deviation (.standardDeviation col)
                            :population-variance (.populationVariance col)
                            :sum (.sum col)
                            :product (.product col)
                            :quartile-1 (.quartile1 col)
                            :quartile-3 (.quartile3 col)
                            )]))
                  (into {})))))

  (column-values [this]
    (when-not (= 0 (dtype/ecount this))
      (or (dtype/->array this)
          (dtype/->array-copy this))))

  (is-missing? [this idx]
    (-> (.isMissing col)
        (.contains (int idx))))

  (get-column-value [this idx]
    (let [idx (int idx)]
      (when (< idx 0)
        (throw (ex-info "Index out of range" {:index idx})))
      (when (>= idx (.size col))
        (throw (ex-info "Index out of range" {:index idx})))
      (if-not (col-proto/is-missing? this idx)
        (.get col (int idx))
        (throw (ex-info (format "Column is missing index %s" idx) {})))))

  (set-values [this idx-val-seq]
    (let [new-col (.copy col)
          col-dtype (dtype/get-datatype col)]
      (doseq [[idx col-val] idx-val-seq]
        (.set new-col (int idx) (col-datatype-cast col-val col-dtype)))
      (->TablesawColumn new-col (select-keys metadata [:categorical? :target?]))))

  (select [this idx-seq]
    (let [^ints int-data (if (instance? (Class/forName "[I") idx-seq)
                           idx-seq
                           (int-array idx-seq))]
      ;;We can't cache much metadata now as we don't really know.
      (->TablesawColumn (.subset col int-data) (select-keys  metadata [:categorical?
                                                                       :target?]))))

  (empty-column [this datatype elem-count column-name]
    (->TablesawColumn
     (dtype-tbl/make-empty-column datatype elem-count {:column-name column-name})
     {}))

  (new-column [this datatype elem-count-or-values column-name]
    (->TablesawColumn
     (dtype-tbl/make-column datatype elem-count-or-values {:column-name column-name})
     {}))

  (math-context [this]
    (->ComputeTensorMathContext))

  dtype-base/PDatatype
  (get-datatype [this] (dtype-base/get-datatype col))

  dtype-base/PContainerType
  (container-type [this] (dtype-base/container-type col))

  dtype-base/PAccess
  (get-value [this idx] (dtype-base/get-value col idx))
  (set-value! [this offset val] (dtype-base/set-value! col offset val))
  (set-constant! [this offset value elem-count]
    (dtype-base/set-constant! col offset value elem-count))

  dtype-base/PCopyRawData
  (copy-raw->item! [raw-data ary-target target-offset options]
    (dtype-base/copy-raw->item! raw-data col target-offset options))

  dtype-base/PPrototype
  (from-prototype [src datatype shape]
    (->TablesawColumn
     (dtype-base/from-prototype col datatype shape)
     {}))

  primitive/PToBuffer
  (->buffer-backing-store [item]
    (primitive/->buffer-backing-store col))

  primitive/POffsetable
  (offset-item [src offset]
    (primitive/offset-item col offset))

  primitive/PToArray
  (->array [src] (primitive/->array col))
  (->array-copy [src] (primitive/->array-copy col))

  mp/PElementCount
  (element-count [item] (mp/element-count col)))


(cpu-typed-buffer/generic-extend-java-type TablesawColumn)



(defn ^tech.tablesaw.io.csv.CsvReadOptions$Builder
  ->csv-builder [^String path & {:keys [separator header? date-format]}]
  (if separator
    (doto (CsvReadOptions/builder path)
      (.separator separator)
      (.header (boolean header?)))
    (doto (CsvReadOptions/builder path)
      (.header (boolean header?)))))


(defn tablesaw-columns->tablesaw-dataset
  [table-name columns]
  (ds-proto/->GenericColumnarDataset table-name
                                     (->> columns
                                          (mapv #(->TablesawColumn % (column->metadata %))))))


(defn ->tablesaw-dataset
  [^Table table]
  (tablesaw-columns->tablesaw-dataset (.name table) (.columns table)))


(defn path->tablesaw-dataset
  [path & {:keys [separator quote]}]
  (-> (Table/read)
      (.csv (->csv-builder path :separator separator :header? true))
      ->tablesaw-dataset))


(defn- in-range
  [[l-min l-max] [r-min r-max]]
  (if (integer? r-min)
    (let [l-min (long l-min)
          l-max (long l-max)
          r-min (long r-min)
          r-max (long r-max)]
      (if (and (>= l-min r-min)
               (<= l-min r-max)
               (>= l-max r-min)
               (<= l-max r-max))
        true
        false))
    (let [l-min (double l-min)
          l-max (double l-max)
          r-min (double r-min)
          r-max (double r-max)]
      (if (and (>= l-min r-min)
               (<= l-min r-max)
               (>= l-max r-min)
               (<= l-max r-max))
        true
        false))))


(defn autoscan-map-seq
  [map-seq-dataset {:keys [scan-depth]
                    :as options}]
  (->> (take 100 map-seq-dataset)
       (reduce (fn [defs next-row]
                 (reduce (fn [defs [row-name row-val]]
                           (let [{:keys [datatype min-val max-val] :as existing}
                                 (get defs row-name {:name row-name})]
                             (assoc defs row-name
                                    (cond
                                      (nil? row-val)
                                      existing

                                      (keyword? row-val)
                                      (assoc existing :datatype :string)

                                      (string? row-val)
                                      (assoc existing :datatype :string)

                                      (number? row-val)
                                      (assoc existing
                                             :min-val (if min-val
                                                        (apply min [min-val row-val])
                                                        row-val)
                                             :max-val (if max-val
                                                        (apply max [max-val row-val])
                                                        row-val)
                                             :datatype (if (integer? row-val)
                                                         (if (= datatype :boolean)
                                                           :boolean
                                                           (or datatype :integer))
                                                         :float))
                                      (boolean? row-val)
                                      (assoc existing
                                             :datatype
                                             (if (#{:integer :float} datatype)
                                               datatype
                                               :boolean))))))
                         defs
                         next-row))
               {})
       ((fn [def-map]
          (->> def-map
               (map (fn [[defname {:keys [datatype min-val max-val] :as definition}]]
                      {:name defname
                       :datatype
                       (if (nil? datatype)
                         :string
                         (case datatype
                           :integer (let [val-range [min-val max-val]]
                                      (cond
                                        (in-range val-range [Short/MIN_VALUE Short/MAX_VALUE])
                                        :int16
                                        (in-range val-range [Integer/MIN_VALUE Integer/MAX_VALUE])
                                        :int32
                                        :else
                                        :int64))
                           :float (let [val-range [min-val max-val]]
                                    (cond
                                      (in-range val-range [(- Float/MAX_VALUE) Float/MAX_VALUE])
                                      :float32
                                      :else
                                      :float64))
                           :string :string
                           :boolean :boolean))})))))))


(defn map-seq->tablesaw-dataset
  [map-seq-dataset & {:keys [scan-depth
                             column-definitions
                             table-name]
                      :or {scan-depth 100
                           table-name "_unnamed"}
                      :as options}]

  (let [column-definitions
        (if column-definitions
          column-definitions
          (autoscan-map-seq map-seq-dataset options))
        ;;force the dataset here as knowing the count helps
        column-map (->> column-definitions
                        (map (fn [{colname :name
                                   datatype :datatype
                                   :as coldef}]
                               [colname
                                (dtype-tbl/make-empty-column datatype 0 {:column-name (->nice-name colname)})]))
                        (into {}))
        all-column-names (set (keys column-map))
        max-idx (reduce (fn [max-idx [idx item-row]]
                          (doseq [[item-name item-val] item-row]
                            (let [^Column col (get column-map item-name)
                                  missing (- (int idx) (.size col))]
                              (dotimes [idx missing]
                                (.appendMissing col))
                              (if-not (nil? item-val)
                                (.append col (col-datatype-cast item-val (dtype/get-datatype col)))
                                (.appendMissing col))))
                          idx)
                        0
                        (map-indexed vector map-seq-dataset))
        column-seq (vals column-map)
        max-ecount (long (if (seq column-seq)
                           (apply max (map dtype/ecount column-seq))
                           0))]
    ;;Ensure all columns are same length
    (doseq [^Column col column-seq]
      (let [missing-count (- max-ecount (.size col))]
        (dotimes [idx missing-count]
          (.appendMissing col))))
    (tablesaw-columns->tablesaw-dataset table-name column-seq)))


(comment
  (def simple-ds (-> (path->tablesaw-dataset "data/aimes-house-prices/train.csv")
                     (etl/apply-pipeline '[[remove "Id"]
                                           ;;Replace missing values or just empty csv values with NA
                                           [replace-string string? "" "NA"]
                                           [replace-missing numeric? 0]
                                           [->etl-datatype numeric?]
                                           [string->number "Utilities" [["NA" -1] "ELO" "NoSeWa" "NoSewr" "AllPub"]]
                                           [string->number "LandSlope" ["Gtl" "Mod" "Sev" "NA"]]
                                           [string->number ["ExterQual"
                                                            "ExterCond"
                                                            "BsmtQual"
                                                            "BsmtCond"
                                                            "HeatingQC"
                                                            "KitchenQual"
                                                            "FireplaceQu"
                                                            "GarageQual"
                                                            "GarageCond"
                                                            "PoolQC"]   ["Ex" "Gd" "TA" "Fa" "Po" "NA"]]
                                           [set-attribute ["MSSubClass" "OverallQual" "OverallCond"] :categorical? true]
                                           [string->number "HasMasVnr" {"BrkCmn" 1
                                                                        "BrkFace" 1
                                                                        "CBlock" 1
                                                                        "Stone" 1
                                                                        "None" 0
                                                                        "NA" -1}]
                                           [string->number "BoughtOffPlan" {"Abnorml" 0
                                                                            "Alloca" 0
                                                                            "AdjLand" 0
                                                                            "Family" 0
                                                                            "Normal" 0
                                                                            "Partial" 1
                                                                            "NA" -1}]
                                           ;;Auto convert the rest that are still string columns
                                           [string->number string?]
                                           [m= "SalePrice" (log1p (col))]]
                                         {:training? true
                                          :target "SalePrice"})))
  )
