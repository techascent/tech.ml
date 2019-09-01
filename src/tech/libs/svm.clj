(ns tech.libs.svm
  (:require [tech.ml.dataset :as dataset]
            [tech.ml.dataset.options :as ds-options]
            [tech.ml.model :as model]
            [tech.ml.protocols.system :as ml-proto]
            [tech.ml.gridsearch :as ml-gs]
            [tech.ml.registry :as registry]
            [tech.ml.model :as model]
            [tech.v2.datatype :as dtype]
            [clojure.string :as s]
            [tech.jna :as jna]
            [tech.v2.datatype.jna :as dtype-jna])
  (:import [tech.libs.svm Types$SVMNode$ByReference Types$SVMProblem$ByReference
            Types$SVMParameter$ByReference
            Types$SVMModel$ByReference
            Types$SVMModel
            Types$SVMNode
            Types$PrintString]
           [com.sun.jna Pointer Native]
           [com.sun.jna.ptr PointerByReference]
           [java.io ByteArrayOutputStream ByteArrayInputStream
            InputStream OutputStream]))


(set! *warn-on-reflection* true)



(def kernel-types {:linear 0, :poly 1, :pre-computed 4, :rbf 2, :sigmoid 3})

(def svm-types {:c-svc 0, :epsilon-svr 3, :nu-svc 1, :nu-svr 4, :one-class 2})

(def default-params
  {:C 1
   :cache-size 100
   :coef0 0
   :degree 3
   :eps 1e-3
   :gamma 0
   :nr-weight 0
   :svm-type (svm-types :c-svc)
   :kernel-type (kernel-types :rbf)
   :nu 0.5
   :p 0.1
   :probability 1
   :shrinking 1})


(defn keyword->kernel-type
  ^long [kwd]
  (if-let [retval (get kernel-types kwd)]
    retval
    (throw (ex-info "Failed to get kernel type"
                    {:kernel-type kwd
                     :possible-types (keys kernel-types)}))))

(defn keyword->svm-type
  [kwd]
  (if-let [retval (get svm-types kwd)]
    retval
    (throw (ex-info "Failed to get kernel type"
                    {:kernel-type kwd
                     :possible-types (keys svm-types)}))))


(defmulti model-type->svm-type
  (fn [model-type]
    model-type))


(defmethod model-type->svm-type :default
  [model-type]
  (if-let [retval (get svm-types model-type)]
    retval
    (throw (ex-info "Failed to find svm type for model type"
                    {:model-type model-type
                     :all-types (keys svm-types)}))))


(defmethod model-type->svm-type :regression
  [_]
  (get svm-types :epsilon-svr))


(defmethod model-type->svm-type :classification
  [_]
  (get svm-types :c-svc))


(defn classification?
  [model-type]
  (#{:classification :c-svc :nu-svc} model-type))


(defn make-params
  "Make trainer params from `dataset` and `options`."
  [options feature-ecount]
  (let [params (Types$SVMParameter$ByReference.)
        model-type (model/options->model-type options)
        svm-type (model-type->svm-type model-type)
        options (-> (update options :kernel-type
                            (fn [k-type]
                              (keyword->kernel-type (or k-type :rbf))))
                    (assoc :svm-type svm-type)
                    ;;For classification we always want to estimate probabilities.
                    (assoc :probability (if (classification? model-type)
                                          1
                                          0)))
        options (update options :gamma
                        #(if (= 0 (long (or % 0)))
                           (/ 1.0 (double feature-ecount))
                           options))]
    (doseq [[key val] (-> (merge default-params options)
                          (select-keys (keys default-params)))]
      (clojure.lang.Reflector/setInstanceField params (s/replace (name key) "-" "_") val))
    (.write params)
    params))


(defn- make-node-array
  ^"[Ltech.libs.svm.Types$SVMNode$ByReference;" [^long num-nodes]
  (let [temp-node (Types$SVMNode$ByReference.)]
    (.toArray temp-node num-nodes)))


(defn- make-nodes
  "Make a LibSVM node."
  ^"[Ltech.libs.svm.Types$SVMNode$ByReference;" [^doubles values]
  (let [num-nodes (alength values)
        ^"[Ltech.libs.svm.Types$SVMNode$ByReference;" nodes (make-node-array (inc num-nodes))]
    (doseq [idx (range num-nodes)]
      (let [^Types$SVMNode$ByReference node (aget nodes idx)]
        (set! (.index node) (inc idx))
        (set! (.value node) (aget values idx))
        (.write node)))
    (let [^Types$SVMNode$ByReference last-node (aget nodes num-nodes)]
      (set! (.index last-node) -1)
      (.write last-node))
    nodes))


(defn- dataset->svm-dataset
  "An svm dataset is an array of indexed nodes along with
  a double-array of 'labels'"
  [row-major-dataset labels?]
  (let [num-items (count row-major-dataset)
        feature-ecount (dtype/ecount (:features (first row-major-dataset)))
        labels (when labels?
                 (dtype/make-container
                  :native-buffer
                  :float64
                  (map (comp #(dtype/get-value % 0)
                             :label) row-major-dataset)))
        nodes (->> row-major-dataset
                   (map :features)
                   (map make-nodes)
                   into-array)]
    (cond-> {:nodes nodes
             :elem-count num-items
             :feature-ecount feature-ecount}
      labels?
      (assoc :labels labels))))


(defn- node-array->pointer-pointer
  [node-array-array]
  (let [ptr-size Native/POINTER_SIZE
        array-datatype (case ptr-size
                         4 :int32
                         8 :int64)]
    (dtype/make-container
     :native-buffer
     array-datatype
     (->> node-array-array
          (map (fn [^"[Ltech.libs.svm.Types$SVMNode$ByReference;" node-ary]
                 (let [^Types$SVMNode$ByReference first-node (aget node-ary 0)]
                   (-> (.getPointer first-node)
                       Pointer/nativeValue))))))))


(jna/def-jna-fn "svm" svm_check_parameter
  "Check if these parameters make sense"
  Pointer
  [problem (partial jna/ensure-type Types$SVMProblem$ByReference)]
  [params (partial jna/ensure-type Types$SVMParameter$ByReference)])

(jna/def-jna-fn "svm" svm_train
  "Train a model"
  Types$SVMModel$ByReference
  [problem (partial jna/ensure-type Types$SVMProblem$ByReference)]
  [params (partial jna/ensure-type Types$SVMParameter$ByReference)])


(jna/def-jna-fn "svm" svm_save_model
  "Save an SVM model"
  Integer/TYPE
  [model-filename str]
  [model (partial jna/ensure-type Types$SVMModel$ByReference)])


(jna/def-jna-fn "svm" svm_load_model
  "Load an SVM model"
  Types$SVMModel$ByReference
  [model-filename str])


(jna/def-jna-fn "svm" svm_predict
  "Predict a value"
  Double/TYPE
  [model (partial jna/ensure-type Types$SVMModel$ByReference)]
  [features (partial jna/ensure-type Types$SVMNode$ByReference)])


(jna/def-jna-fn "svm" svm_predict_probability
  "Predict and return probability estimates"
  Double/TYPE
  [model (partial jna/ensure-type Types$SVMModel$ByReference)]
  [features (partial jna/ensure-type Types$SVMNode$ByReference)]
  [prob-estimates (partial jna/ensure-type (Class/forName "[D"))])


(jna/def-jna-fn "svm" svm_free_and_destroy_model
  "Free a loaded model"
  Integer
  [model jna/ensure-ptr-ptr])


(jna/def-jna-fn "svm" svm_set_print_string_function
  "Set the print function"
  nil
  [callback (partial jna/ensure-type Types$PrintString)])

(def no-print-printer (proxy [Types$PrintString] []
                          (Print [^String data]
                                 ;;don't print!!
                            )))

(svm_set_print_string_function no-print-printer)


(defrecord SVMSystem []
  ml-proto/PMLSystem
  (system-name [_] :libsvm)

  (gridsearch-options [system options]
    ;;These are just a very small set of possible options
    {:kernel-type (ml-gs/nominative [:linear :poly :rbf :sigmoid])
     :C (ml-gs/exp [1e-3 1e4])
     :eps (ml-gs/exp [1e-8 1e-1])})

  (train [system options dataset]
    (let [row-major-dataset (dataset/->row-major dataset options)
          {:keys [nodes elem-count feature-ecount labels]}
          (dataset->svm-dataset row-major-dataset true)
          problem (Types$SVMProblem$ByReference.)
          params (make-params options feature-ecount)
          node-pointer-pointer-buffer (node-array->pointer-pointer nodes)
          gc-root [nodes node-pointer-pointer-buffer labels]]
      (set! (.l problem) (int elem-count))
      (set! (.y problem) ^Pointer (jna/->ptr-backing-store labels))
      (set! (.x problem) ^Pointer (jna/->ptr-backing-store node-pointer-pointer-buffer))
      (.write problem)
      (let [problem-description (svm_check_parameter problem params)]
        (when-not (= 0 (Pointer/nativeValue problem-description))
          (throw (ex-info (format "SVM check error: %s" (jna/variable-byte-ptr->string
                                                         problem-description))
                          {}))))
      (let [^Types$SVMModel$ByReference model (svm_train problem params)
            ;;SVM save is not reentrant
            retval (locking system
                     (let [retval
                           (model/model-file-save->byte-array #(svm_save_model % model))]
                       retval))]
        (get gc-root 0)
        (svm_free_and_destroy_model (PointerByReference. (.getPointer model)))
        retval)))

  (thaw-model [system model]
    ;;SVM load is not reentrant
    (locking system
      (let [retval
            (model/byte-array-file-load->model model svm_load_model)]
        retval)))

  (predict [system options thawed-model dataset]
    (let [row-major-dataset (dataset/->row-major dataset options)
          ^Types$SVMModel$ByReference model thawed-model
          {:keys [nodes]} (dataset->svm-dataset row-major-dataset false)]
      (locking model
        (if (classification? (model/options->model-type options))
          (let [label-map (ds-options/inference-target-label-map options)
                pre-ordered-labels (->> label-map
                                        (sort-by second)
                                        (mapv first))
                model-labels (-> (dtype-jna/unsafe-address->typed-pointer
                                  (Pointer/nativeValue (.label model))
                                  (* (dtype/datatype->byte-size :int32)
                                     (.nr_class model))
                                  :int32))
                model-labels (dtype/->vector model-labels)
                ;;SVM reorders labels
                ordered-labels (mapv pre-ordered-labels model-labels)
                probabilities (double-array (.nr_class model))
                retval (->> nodes
                            (mapv (fn [^"[Ltech.libs.svm.Types$SVMNode$ByReference;" feature]
                                    (let [first-node (aget feature 0)
                                          prediction (svm_predict_probability model first-node probabilities)]
                                      (zipmap ordered-labels (vec probabilities))))))]
            retval)
          (let [retval
                (->> nodes
                     (mapv (fn [^"[Ltech.libs.svm.Types$SVMNode$ByReference;" feature]
                             (let [first-node (aget feature 0)]
                               (svm_predict model first-node)))))]
            retval))))))


(def system (constantly (->SVMSystem )))


(registry/register-system (system))
