(ns tech.ml.dataset.etl.column-filters
  (:require [tech.ml.protocols.dataset :as ds-proto]
            [tech.ml.protocols.column :as col-proto]
            [tech.ml.protocols.etl :as etl-proto]
            [tech.datatype.java-unsigned :as unsigned]
            [clojure.set :as c-set]))


(defonce ^:dynamic *column-filter-fns* (atom {}))


(defn register-column-filter!
  [name-kwd filter-fn]
  (-> (swap! *column-filter-fns* assoc name-kwd filter-fn)
      keys))


(defn- execute-col-filter-fn
  [dataset filter-fn args]
  (if-let [filter-op (get @*column-filter-fns* filter-fn)]
    (apply filter-op dataset args)
    (throw (ex-info (format "Failed to find column filter fn: %s"
                            filter-fn)
                    {:filter-fn filter-fn}))))


(defn execute-column-filter
  [dataset col-filter]
  (cond
    (sequential? col-filter)
    (let [filter-fn (-> (first col-filter)
                        name
                        keyword)]
      (execute-col-filter-fn dataset filter-fn (rest col-filter)))
    (or (symbol? col-filter) (keyword? col-filter))
    (execute-col-filter-fn dataset (keyword (name col-filter)) [])))


(defn select-columns
  [dataset col-selector]
  (cond
    (string? col-selector)
    [col-selector]
    (sequential? col-selector)
    (if (string? (first col-selector))
      col-selector
      (execute-column-filter dataset col-selector))
    (symbol? col-selector)
    (execute-column-filter dataset col-selector)))


(defn- process-filter-args
  [dataset args]
  (if (seq args)
    (map (partial ds-proto/column dataset) args)
    (ds-proto/columns dataset)))


(defn- basic-metadata-filter
  [filter-fn dataset & args]
  (->> (process-filter-args dataset args)
       (map (comp col-proto/metadata))
       (filter filter-fn)
       (map :name)))


(defn register-exact-datatype-filter
  [dtype-kwd]
  (let [filter-kwd (keyword (str (name dtype-kwd) "?"))]
    (register-column-filter!
     filter-kwd
     (partial basic-metadata-filter #(= dtype-kwd (:datatype %))))))


(doseq [dtype-kwd (concat [:boolean :string]
                          unsigned/datatypes)]
  (register-exact-datatype-filter dtype-kwd))


(register-column-filter!
 :numeric?
 (partial basic-metadata-filter #((set unsigned/datatypes) (:datatype %))))


(register-column-filter!
 :not
 (fn [dataset & args]
   (let [all-names (set (map col-proto/column-name (ds-proto/columns dataset)))]
     (c-set/difference all-names (set (execute-column-filter dataset (first args)))))))


(register-column-filter!
 :or
 (fn [dataset & args]
   (apply c-set/union #{} (map (comp set (partial execute-column-filter dataset))
                               args))))


(register-column-filter!
 :and
 (fn [dataset & args]
   (let [live-set (set (execute-column-filter dataset (first args)))]
      (->> (rest args)
           (reduce (fn [live-set arg]
                     (when-not (= 0 (count live-set))
                       (c-set/intersection live-set (set (execute-column-filter dataset arg)))))
                   live-set)))))


(register-column-filter!
 :categorical?
 (partial basic-metadata-filter :categorical?))

(register-column-filter!
 :target?
 (partial basic-metadata-filter :target?))


(register-column-filter!
 :*
 (fn [dataset & args]
   (->> (process-filter-args args)
        (map col-proto/column-name))))
