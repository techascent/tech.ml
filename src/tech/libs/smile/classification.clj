(ns tech.libs.smile.classification
  (:require [tech.v2.datatype :as dtype]
            [tech.v2.datatype.casting :as casting]
            [tech.ml.protocols.system :as ml-proto]
            [tech.ml.model :as model]
            [tech.ml.registry :as registry]
            [tech.ml.dataset :as ds]
            [tech.ml.gridsearch :as ml-gs]
            [tech.ml.util :as ml-util]
            [tech.ml.dataset.options :as ds-options]
            [tech.libs.smile.protocols :as smile-proto])
  (:import [smile.classification SoftClassifier AdaBoost]
           [smile.data.formula Formula]))


(set! *warn-on-reflection* true)


(defn tuple-predict-posterior
  [^SoftClassifier model ds options n-labels]
  (let [df (ds/dataset->smile-dataframe ds)]
    (smile-proto/initialize-model-formula! model options)
    (->> (dtype/make-reader
          :posterior-probabilities
          (ds/row-count ds)
          (let [posterior (double-array n-labels)]
            (.predict model (.get df idx) posterior)
            posterior))
         (dtype/make-container :typed-buffer :posterior-probabilities))))


(def classifier-metadata
  {:ada-boost {:name :ada-boosts
               :options [{:name :trees
                          :type :int32
                          :default 500}
                         {:name :max-depth
                          :type :int32
                          :default 200}
                         {:name :max-nodes
                          :type :int32
                          :default 6}
                         {:name :node-size
                          :type :int32
                          :default 1}]
               :gridsearch-options {:trees (ml-gs/linear-long [2 500])
                                    :max-nodes (ml-gs/linear-long [4 1000])}
               :property-name-stem "smile.databoost"
               :constructor #(AdaBoost/fit ^Formula %1 ^DataFrame %2 ^Properties %3)
               :predictor tuple-predict-posterior}

   ;; :decision-tree {:attributes #{:probabilities :attributes}
   ;;                 :class-name "DecisionTree"
   ;;                 :datatypes #{:float64-array}
   ;;                 :name :decision-tree
   ;;                 :options [{:name :max-nodes
   ;;                            :type :int32
   ;;                            :default 100}
   ;;                           {:name :node-size
   ;;                            :type :int32
   ;;                            :default 1}
   ;;                           {:name :split-rule
   ;;                            :type :enumeration
   ;;                            :class-type DecisionTree$SplitRule
   ;;                            :lookup-table {:gini DecisionTree$SplitRule/GINI
   ;;                                           :entropy DecisionTree$SplitRule/ENTROPY
   ;;                                           :classification-error DecisionTree$SplitRule/CLASSIFICATION_ERROR}
   ;;                            :default :gini}]}
   ;; :fld {:attributes #{:projection}
   ;;       :class-name "FLD"
   ;;       :datatypes #{:float64-array}
   ;;       :name :fld
   ;;       :options [{:name :L
   ;;                  :type :int32
   ;;                  :default -1}
   ;;                 {:name :tolerance
   ;;                  :type :float64
   ;;                  :default 1e-4}]}
   ;; :gradient-tree-boost {:attributes #{:probabilities}
   ;;                       :class-name "GradientTreeBoost"
   ;;                       :datatypes #{:float64-array}
   ;;                       :name :gradient-tree-boost
   ;;                       :options [{:name :ntrees
   ;;                                  :type :int32
   ;;                                  :default 500}
   ;;                                 {:name :max-nodes
   ;;                                  :type :int32
   ;;                                  :default 6
   ;;                                  :range :>0}
   ;;                                 {:name :shrinkage
   ;;                                  :type :float64
   ;;                                  :default 0.005
   ;;                                  :range :>0}
   ;;                                 {:name :sampling-fraction
   ;;                                  :type :float64
   ;;                                  :default 0.7
   ;;                                  :range [0.0 1.0]}]}
   ;; :knn {:attributes #{:probabilities :object-data}
   ;;       :class-name "KNN"
   ;;       :datatypes #{:float64-array}
   ;;       :name :knn
   ;;       :options [{:name :distance
   ;;                  :type :distance
   ;;                  :default {:distance-type :euclidean}}
   ;;                 {:name :num-clusters
   ;;                  :type :int32
   ;;                  :default 5}]
   ;;       :gridsearch-options {:num-clusters (ml-gs/linear-long [2 100])}}
   ;; :logistic-regression {:attributes #{:probabilities}
   ;;                       :class-name "LogisticRegression"
   ;;                       :datatypes #{:float64-array}
   ;;                       :name :logistic-regression
   ;;                       :options [{:name :lambda
   ;;                                  :type :float64
   ;;                                  :default 0.1}
   ;;                                 {:name :tolerance
   ;;                                  :type :float64
   ;;                                  :default 1e-5}
   ;;                                 {:name :max-iter
   ;;                                  :type :int32
   ;;                                  :default 500}]
   ;;                       :gridsearch-options {:lambda (ml-gs/exp [1e-3 1e2])
   ;;                                            :tolerance (ml-gs/linear [1e-9 1e-1])
   ;;                                            :max-iter (ml-gs/linear-long [1e2 1e4])}}
   ;; ;;Not supported at this time because constructor patter is unique
   ;; :maxent {:attributes #{:probabilities}
   ;;          :class-name "Maxent"
   ;;          :datatypes #{:float64-array :int32-array}
   ;;          :name :maxent}

   ;; :naive-bayes {:attributes #{:online :probabilities}
   ;;               :class-name "NaiveBayes"
   ;;               :datatypes #{:float64-array :sparse}
   ;;               :name :naive-bayes
   ;;               :options [{:name :model
   ;;                          :type :enumeration
   ;;                          :class-type NaiveBayes$Model
   ;;                          :lookup-table {
   ;;                                         ;; Users have to provide probabilities for this to work.
   ;;                                         ;; :general NaiveBayes$Model/GENERAL

   ;;                                         :multinomial NaiveBayes$Model/MULTINOMIAL
   ;;                                         :bernoulli NaiveBayes$Model/BERNOULLI
   ;;                                         :polyaurn NaiveBayes$Model/POLYAURN}
   ;;                          :default :multinomial}
   ;;                         {:name :num-classes
   ;;                          :type :int32
   ;;                          :default utils/options->num-classes}
   ;;                         {:name :input-dimensionality
   ;;                          :type :int32
   ;;                          :default utils/options->feature-ecount}
   ;;                         {:name :sigma
   ;;                          :type :float64
   ;;                          :default 1.0}]
   ;;               :gridsearch-options {:model (ml-gs/nominative [:multinomial :bernoulli :polyaurn])
   ;;                                    :sigma (ml-gs/exp [1e-4 0.2])}}
   ;; :neural-network {:attributes #{:online :probabilities}
   ;;                  :class-name "NeuralNetwork"
   ;;                  :datatypes #{:float64-array}
   ;;                  :name :neural-network}
   ;; :platt-scaling {:attributes #{}
   ;;                 :class-name "PlattScaling"
   ;;                 :datatypes #{:double}
   ;;                 :name :platt-scaling}


   ;; ;;Lots of discriminant analysis
   ;; :linear-discriminant-analysis
   ;; {:attributes #{:probabilities}
   ;;  :class-name "LDA"
   ;;  :datatypes #{:float64-array}
   ;;  :name :lda
   ;;  :options [{:name :prioiri
   ;;             :type :float64-array
   ;;             :default nil}
   ;;            {:name :tolerance
   ;;             :default 1e-4
   ;;             :type :float64}]
   ;;  :gridsearch-options {:tolerance (ml-gs/linear [1e-9 1e-2])}}


   ;; :quadratic-discriminant-analysis
   ;; {:attributes #{:probabilities}
   ;;  :class-name "QDA"
   ;;  :datatypes #{:float64-array}
   ;;  :name :qda
   ;;  :options [{:name :prioiri
   ;;             :type :float64-array
   ;;             :default nil}
   ;;            {:name :tolerance
   ;;             :default 1e-4
   ;;             :type :float64}]
   ;;  :gridsearch-options {:tolerance (ml-gs/linear [1e-9 1e-2])}}


   ;; :regularized-discriminant-analysis
   ;; {:attributes #{:probabilities}
   ;;  :class-name "RDA"
   ;;  :datatypes #{:float64-array}
   ;;  :name :rda
   ;;  :options [{:name :prioiri
   ;;             :type :float64-array
   ;;             :default nil}
   ;;            {:name :alpha
   ;;             :type :float64
   ;;             :default 0.0 }
   ;;            {:name :tolerance
   ;;             :default 1e-4
   ;;             :type :float64}]
   ;;  :gridsearch-options {:tolerance (ml-gs/linear [1e-9 1e-2])
   ;;                       :alpha (ml-gs/linear [0.0 1.0])}}


   ;; :random-forest {:attributes #{:probabilities}
   ;;                 :class-name "RandomForest"
   ;;                 :datatypes #{:float64-array}
   ;;                 :name :random-forest}
   ;; :rbf-network {:attributes #{}
   ;;               :class-name "RBFNetwork"
   ;;               :datatypes #{}
   ;;               :name :rbf-network}


   ;; :svm {:attributes #{:online :probabilities}
   ;;       :class-name "SVM"
   ;;       :datatypes #{:float64-array}
   ;;       :name :svm
   ;;       :constructor-filter (fn [options mixed-data-entry]
   ;;                             ;;There is a different constructor when the number of classes is 2
   ;;                             (if (> (utils/options->num-classes options)
   ;;                                    2)
   ;;                               mixed-data-entry
   ;;                               (let [opt-name (-> (nth mixed-data-entry 2)
   ;;                                                  :name)]
   ;;                                 (when-not (or (= opt-name :multiclass-strategy)
   ;;                                               (= opt-name :num-classes))
   ;;                                     mixed-data-entry))))
   ;;       :options [{:name :kernel
   ;;                  :type :mercer-kernel
   ;;                  :default {:kernel-type :gaussian}}
   ;;                 {:name :soft-margin-penalty
   ;;                  :type :float64
   ;;                  :altname "C"
   ;;                  :default 1.0}
   ;;                 {:name :num-classes
   ;;                  :type :int32
   ;;                  :default utils/options->num-classes}
   ;;                 {:name :multiclass-strategy
   ;;                  :type :enumeration
   ;;                  :class-type SVM$Multiclass
   ;;                  :lookup-table {:one-vs-one SVM$Multiclass/ONE_VS_ONE
   ;;                                 :one-vs-all SVM$Multiclass/ONE_VS_ALL}
   ;;                  :default :one-vs-one}]
   ;;       :gridsearch-options {:kernel {:kernel-type (ml-gs/nominative [:gaussian :linear])}
   ;;                            :soft-margin-penalty (ml-gs/exp [1e-4 1e2])
   ;;                            :multiclass-strategy (ml-gs/nominative [:one-vs-one
   ;;                                                                    :one-vs-all])}}
   })


(defmulti model-type->classification-model
  (fn [model-type] model-type))


(defmethod model-type->classification-model :default
  [model-type]
  (if-let [retval (get classifier-metadata model-type)]
    retval
    (throw (ex-info "Failed to find classification model"
                    {:model-type model-type
                     :available-types (keys classifier-metadata)}))))


(defrecord SmileClassification []
  ml-proto/PMLSystem
  (system-name [_] :smile.classification)
  (gridsearch-options [system options]
    (let [entry-metadata (model-type->classification-model
                          (model/options->model-type options))]
      (if-let [retval (:gridsearch-options entry-metadata)]
        retval
        (throw (ex-info "Model type does not support auto gridsearch yet"
                        {:entry-metadata entry-metadata})))))
  (train [system options dataset]
    (let [entry-metadata (model-type->classification-model
                          (model/options->model-type options))
          target-colname (:target options)
          formula (Formula. (ml-util/->str target-colname))
          dataset (if (casting/integer-type? (dtype/get-datatype
                                              (dataset target-colname)))
                    dataset
                    (ds/column-cast dataset target-colname :int32))
          data (ds/dataset->smile-dataframe dataset)
          properties (smile-proto/options->properties entry-metadata dataset options)
          ctor (:constructor entry-metadata)
          model (ctor formula data properties)]
      (model/model->byte-array model)))
  (thaw-model [system model]
    (model/byte-array->model model))
  (predict [system options thawed-model dataset]
    (let [entry-metadata (model-type->classification-model
                          (model/options->model-type options))
          target-colname (:target options)
          n-labels (-> (get-in options [:label-map target-colname])
                       count)
          predictor (:predictor entry-metadata)]
      (predictor thawed-model dataset options n-labels))))



(def system (constantly (->SmileClassification)))


(registry/register-system (system))
