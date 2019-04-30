(ns tech.libs.smile.utils
  (:require [tech.v2.datatype :as dtype]
            [tech.ml.dataset :as dataset]
            [tech.ml.dataset.options :as ds-options]
            [tech.ml.utils :as ml-utils])

  (:import [java.lang.reflect Constructor]
           [smile.data Attribute NominalAttribute NumericAttribute]))


(def datatype-map
  {:double<> :float64-array
   :float<> :float32-array
   :int<> :int32-array
   :char<> :char-array
   :java.lang.String :string
   :smile.math.SparseArray :sparse
   :long :int64
   :long<> :int64-array
   :int :int32
   :byte :int8
   :byte<> :int8-array
   :short :int16
   :short<> :int16-array
   :java.util.BitSet :bit-set
   :java.lang.Object<> :object-array
   })


(defn method-datatype
  [methodname reflect-data]
  (->> reflect-data
       :members
       (filter #(= methodname (name (:name %))))
       (mapcat :parameter-types)
       (map (comp keyword name))
       (remove #(= :java.lang.Object %))
       (map #(get datatype-map % %))
       set))


(def keyword->class-types
  {:float64 Double/TYPE
   :int32 Integer/TYPE
   :boolean Boolean/TYPE
   :int32-array (Class/forName "[I")
   :float64-array (Class/forName "[D")
   :float64-array-array (Class/forName "[[D")
   :object-array (Class/forName "[Ljava.lang.Object;")
   :attribute-array (Class/forName "[Lsmile.data.Attribute;")})


(defmulti option->class-type
  (fn [option]
    (:type option)))

(defmethod option->class-type :default
  [option]
  (if-let [retval (keyword->class-types (:type option))]
    retval
    (throw (ex-info "Failed to find primitive class type"
                    {:cls-type (:type option)
                     :available (keys keyword->class-types)}))))


(defmethod option->class-type :enumeration
  [option]
  (:class-type option))


(defn class-name->class
  ^Class [pkgname ^String clsname]
  (Class/forName
   (if-not (.contains clsname ".")
     (str pkgname "." clsname)
     clsname)))


(defmulti option-value->value
  (fn [class-metadata meta-option option-val]
    (:type meta-option)))


(defmethod option-value->value :default
  [class-metadata option-desc option-val]
  (dtype/cast option-val (:type option-desc)))


(defmethod option-value->value :enumeration
  [class-metadata option-desc option-val]
  (if-let [retval (get-in option-desc [:lookup-table option-val])]
    retval
    (throw (ex-info "Failed to find option"
                    {:option-descriptor option-desc
                     :option-value option-val}))))


(defmethod option-value->value :boolean
  [class-metadata option-desc option-val]
  (boolean option-val))


(defmethod option-value->value :int32-array [class-metadata _ val] val)
(defmethod option-value->value :float64-array [class-metadata _ val] val)
(defmethod option-value->value :float64-array-array [class-metadata _ val] val)
(defmethod option-value->value :object-array [class-metadata _ val] val)
(defmethod option-value->value :attribute-array [class-metadata _ val] val)
(defmethod option-value->value :int32-array
  [class-metadata option-desc option-val]
  (let [option-val (if (number? option-val)
                     [option-val]
                     option-val)]
    (dtype/make-array-of-type :int32 option-val)))


(defn metadata-option-default-value
  [metadata-option options]
  (let [option-val (get metadata-option :default)]
    (cond
      (number? option-val)
      option-val
      (nil? option-val)
      option-val
      (map? option-val)
      option-val
      (keyword? option-val)
      option-val
      ;;Some defaults need data about the dataset
      (instance? clojure.lang.IFn option-val)
      (option-val options)
      :else
      option-val)))


(defn- process-class-arguments
  "Process constructor arguments return a map of:
{:class-types - sequence of class types
 :option-values - sequence of option values }"
  [class-metadata options]
  (let [metadata-options (:options class-metadata)
        mixed-data (->> metadata-options
                        (map (fn [meta-option]
                               (let [default-value (metadata-option-default-value
                                                    meta-option options)
                                     found-default? (contains? meta-option :default)
                                     optional? (contains? (:attributes meta-option) :optional?)
                                     option-value (let [option-data (get options (:name meta-option)
                                                                         default-value)]
                                                    (when-not (nil? option-data)
                                                      (option-value->value class-metadata meta-option option-data)))]
                                 (if found-default?
                                   [(option->class-type meta-option) option-value meta-option]
                                   (when-not optional?
                                     (throw (ex-info "Option is neither provided nor optional"
                                                     {:meta-option meta-option
                                                      :options options})))))))
                        (remove nil?))
        setter-filter (fn [[cls-type option-val meta-option]]
                        (:setter meta-option))
        ;;Options that can be set in the constructor should be.
        constructor-data (->> mixed-data
                              (remove setter-filter)
                              (filter (fn [mixed-data-entry]
                                        (if-let [cons-filter (:constructor-filter class-metadata)]
                                          (do
                                            (cons-filter options mixed-data-entry))
                                          mixed-data-entry)))
                              seq)
        ;;Not all options can be set in the constructor.
        setter-data (->> mixed-data
                         (filter setter-filter)
                         seq)]
    (merge {}
           (when constructor-data
             {:constructor-data {:class-types (map first constructor-data)
                                 :values (map second constructor-data)
                                 :metadata (map #(nth % 2) constructor-data)}})
           (when setter-data
             {:setters (->> setter-data
                            (map (fn [[cls-type data-val meta-option]]
                                   {:data data-val
                                    :class-type cls-type
                                    :setter (:setter meta-option)})))}))))


(defn set-setter-options!
  [item setter-data-seq]
  (let [model-cls (.getClass item)]
    (->> setter-data-seq
         (map (fn [{:keys [data class-type setter]}]
                (let [method (.getMethod model-cls setter (into-array ^Class [class-type]))]
                  (.invoke method item (object-array [data])))))
         dorun)))


(defn prepend-data-constructor-arguments
  "Prepend the attributes to the constructor arguments.  Return new entry metadata."
  [entry-metadata options data-constructor-arguments]
  (let [data-constructor-arguments
        ;;Do the attributes of the metadata indicate that the constructor arguments
        ;;require the first argument to be an attribute array describing the data.
        (if (contains? (:attributes entry-metadata) :attributes)
          (let [attribute-array
                (->> (:feature-columns options)
                     (mapcat (fn [feature-key]
                               (let [nominal? (boolean
                                               (get-in options [:dataset-column-metadata :post-pipeline
                                                                feature-key :categorical?]))
                                     att-name (ml-utils/column-safe-name feature-key)
                                     attribute (if nominal?
                                                 (NominalAttribute. att-name)
                                                 (NumericAttribute. att-name))]
                                 [attribute])))
                     (into-array Attribute))]
            (concat [{:type :attribute-array
                      :default attribute-array
                      :name :attributes}]
                    data-constructor-arguments))
          data-constructor-arguments)]
    (update entry-metadata :options
            (fn [opt-list]
              (concat data-constructor-arguments
                      opt-list)))))


(defn construct
  [class-metadata package-name options]
  (let [{:keys [constructor-data setters]} (process-class-arguments class-metadata options)
        full-class (class-name->class package-name (:class-name class-metadata))
        ^Constructor constructor (if constructor-data
                                   (.getConstructor full-class (into-array ^Class (:class-types constructor-data)))
                                   (.getConstructor full-class nil))
        arguments (->> (:values constructor-data)
                       object-array)
        retval (try
                 (.newInstance constructor arguments)
                 (catch Throwable e
                   (throw (ex-info "Failed to construct class"
                                   {:class-metadata class-metadata
                                    :options options
                                    :error e}))))]
    (when setters
      (set-setter-options! retval setters))
    retval))

(defn get-option-value
  [metadata option-name options]
  (get options option-name
       (->> metadata
            :options
            (group-by :name)
            option-name
            first
            :default)))


(defn options->num-classes
  ^long [options]
  (ds-options/num-classes options))


(defn options->feature-ecount
  ^long [options]
  (ds-options/feature-ecount options))
