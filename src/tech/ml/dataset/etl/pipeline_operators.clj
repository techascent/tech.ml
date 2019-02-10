(ns tech.ml.dataset.etl.pipeline-operators
  (:require [tech.ml.dataset :as ds]
            [tech.ml.dataset.column :as ds-col]
            [tech.ml.protocols.etl :as etl-proto]
            [tech.datatype :as dtype]
            [tech.ml.dataset.etl.column-filters :as column-filters]
            [tech.ml.dataset.etl.defaults :refer [etl-datatype]]
            [tech.ml.dataset.etl.math-ops :as math-ops])
  (:refer-clojure :exclude [remove])
  (:import [tech.ml.protocols.etl PETLSingleColumnOperator]))


(defonce ^:dynamic *etl-operator-registry* (atom {}))


(defn register-etl-operator!
  [op-kwd op]
  (-> (swap! *etl-operator-registry* assoc op-kwd op)
      keys))


(defn get-etl-operator
  [op-kwd]
  (if-let [retval (get @*etl-operator-registry* op-kwd)]
    retval
    (throw (ex-info (format "Failed to find etl operator %s" op-kwd)
                    {:op-keyword op-kwd}))))


(defn apply-pipeline-operator
  [{:keys [inference?] :as options} {:keys [pipeline dataset]} op]
  (let [[op context] (if-not inference?
                       [op {}]
                       [(:operation op) (:context op)])
        op-type (keyword (name (first op)))
        col-selector (second op)
        op-args (drop 2 op)
        col-seq (column-filters/select-columns dataset col-selector)
        op-impl (get-etl-operator op-type)
        context (if-not inference?
                  (etl-proto/build-etl-context-columns
                   op-impl dataset col-seq op-args)
                  context)
        dataset (etl-proto/perform-etl-columns
                 op-impl dataset col-seq op-args context)]
    {:dataset dataset
     :pipeline (conj pipeline {:operation (->> (concat [(first op) col-seq]
                                                       (drop 2 op))
                                               vec)
                               :context context})}))



(defmacro def-etl-operator
  [op-symbol op-context-code op-code]
  `(do (register-etl-operator! ~(keyword (name op-symbol))
                               (reify PETLSingleColumnOperator
                                 (build-etl-context [~'op ~'dataset ~'column-name ~'op-args]
                                   ~op-context-code)
                                 (perform-etl [~'op ~'dataset ~'column-name ~'op-args ~'context]
                                   ~op-code)))
       (defn ~op-symbol
         [dataset# col-selector# & op-args#]
         (-> (apply-pipeline-operator {}
                                      {:pipeline []
                                       :dataset dataset#}
                                      (-> (concat '[~op-symbol]
                                                  [col-selector#]
                                                  op-args#)
                                          vec))
             :dataset))))


(def-etl-operator
  set-attribute
  nil
  (let [retval (ds/update-column
                dataset column-name
                (fn [col]
                  (->> (merge (ds-col/metadata col)
                              (apply hash-map op-args))
                       (ds-col/set-metadata col))))]
    retval))


(def-etl-operator
  remove
  nil
  (ds/remove-column dataset column-name))


(def-etl-operator
  replace-missing
  (let [missing-val (first op-args)]
    {:missing-value missing-val})
  (ds/update-column
   dataset column-name
   (fn [col]
     (let [missing-indexes (ds-col/missing col)]
       (ds-col/set-values col (map vector
                                      (seq missing-indexes)
                                      (repeat (:missing-value context))))))))


(defn- make-string-table-from-args
  [op-args]
  (let [used-indexes (->> op-args
                          (map (fn [item]
                                 (if (string? item)
                                   nil
                                   (second item))))
                          (clojure.core/remove nil?)
                          set)]
    (->> op-args
         (reduce (fn [[lookup-table used-indexes] item]
                   (if (string? item)
                     (let [next-idx (first (clojure.core/remove used-indexes (range)))]
                       [(assoc lookup-table item next-idx)
                        (conj used-indexes next-idx)])
                     (let [[item-name item-idx] item]
                       [(assoc lookup-table item-name item-idx)
                        (conj used-indexes item-idx)])))
                 [{} used-indexes])
         first)))


(def-etl-operator
  string->number
  (if-let [table-vals (seq (first op-args))]
    (make-string-table-from-args table-vals)
    (make-string-table-from-args (ds-col/unique (ds/column
                                                    dataset column-name))))
  (ds/update-column
   dataset column-name
   (fn [col]
     (let [existing-values (ds-col/column-values col)
           str-table context
           new-col-dtype (etl-datatype)
           data-values (dtype/make-array-of-type
                        new-col-dtype
                        (->> existing-values
                             (map (fn [item-val]
                                    (if-let [lookup-val (get str-table item-val)]
                                      lookup-val
                                      (throw (ex-info (format "Failed to find lookup for value %s"
                                                              item-val)
                                                      {:item-value item-val
                                                       :possible-values (set (keys str-table))}))))))
                        {:unchecked? true})]
       (-> (ds-col/new-column col new-col-dtype data-values column-name)
           (ds-col/set-metadata (select-keys (ds-col/metadata col)
                                                [:target? :categorical?])))))))


(def-etl-operator
  replace-string
  nil
  (ds/update-column
   dataset column-name
   (fn [col]
     (let [existing-values (ds-col/column-values col)
           [src-str replace-str] op-args
           data-values (into-array String (->> existing-values
                                            (map (fn [str-value]
                                                   (if (= str-value src-str)
                                                     replace-str
                                                     str-value)))))]
       (-> (ds-col/new-column col :string data-values column-name)
           (ds-col/set-metadata (select-keys (ds-col/metadata col)
                                                [:target? :categorical?])))))))


(def-etl-operator
  ->etl-datatype
  nil
  (ds/update-column
   dataset column-name
   (fn [col]
     (if-not (= (dtype/get-datatype col) (etl-datatype))
       (let [new-col-dtype (etl-datatype)
             col-values (ds-col/column-values col)
             data-values (dtype/make-array-of-type
                          new-col-dtype
                          (if (= :boolean (dtype/get-datatype col))
                            (map #(if % 1 0) col-values)
                            col-values))]
         (-> (ds-col/new-column col new-col-dtype data-values column-name)
             (ds-col/set-metadata (select-keys (ds-col/metadata col)
                                                  [:target? :categorical?]))))
       col))))


(defn eval-expr
  "Tiny simple interpreter."
  [{:keys [dataset column-name] :as env} math-expr]
  (cond
    (string? math-expr)
    math-expr
    (number? math-expr)
    math-expr
    (sequential? math-expr)
    (let [fn-name (first math-expr)
          ;;Force errors early
          expr-args (mapv (partial eval-expr env) (rest math-expr))
          {op-type :type
           operand :operand} (math-ops/get-operand (keyword (name fn-name)))]
      (apply operand env expr-args))))


(def-etl-operator
  m=
  nil
  (let [result (eval-expr {:dataset dataset
                           :column-name column-name}
                          (first op-args))
        src-col (ds/maybe-column dataset column-name)]

    (ds/add-or-update-column dataset (cond-> (ds-col/set-name result column-name)
                                             src-col
                                             ;;We can't set anything else as we don't know if the column is categorical
                                             ;;or not  If it was the target, however, it still is.
                                             (ds-col/set-metadata (select-keys (ds-col/metadata src-col)
                                                                                  [:target?]))))))
