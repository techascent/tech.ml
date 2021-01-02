(ns tech.v3.libs.smile.classification
  "Namespace to require to enable a set of smile classification models."
  (:require [tech.v3.datatype :as dtype]
            [tech.v3.datatype.protocols :as dtype-proto]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.utils :as ds-utils]
            [tech.v3.tensor :as dtt]
            [tech.v3.ml.gridsearch :as ml-gs]
            [tech.v3.ml.model :as model]
            [tech.v3.ml :as ml]
            [tech.v3.libs.smile.protocols :as smile-proto]
            [tech.v3.libs.smile.data :as smile-data])
  (:import [smile.classification SoftClassifier AdaBoost LogisticRegression DecisionTree]
           [smile.base.cart SplitRule]
           [smile.data.formula Formula]
           [smile.data DataFrame]
           [java.util Properties List]
           [tech.v3.datatype ObjectReader]))


(set! *warn-on-reflection* true)

(defn- tuple-predict-posterior
  [^SoftClassifier model ds options n-labels]
  (let [df (smile-data/dataset->smile-dataframe ds)
        n-rows (ds/row-count ds)]
    (smile-proto/initialize-model-formula! model ds)
    (reify
      dtype-proto/PShape
      (shape [rdr] [n-rows n-labels])
      ObjectReader
      (lsize [rdr] n-rows)
      (readObject [rdr idx]
        (let [posterior (double-array n-labels)]
          (.predict model (.get df idx) posterior)
          posterior)))))


(defn- double-array-predict-posterior
  [^SoftClassifier model ds options n-labels]
  (let [value-reader (ds/value-reader ds)
        n-rows (ds/row-count ds)]
    (reify
      dtype-proto/PShape
      (shape [rdr] [n-rows n-labels])
      ObjectReader
      (lsize [rdr] n-rows)
      (readObject [rdr idx]
        (let [posterior (double-array n-labels)]
          (.predict model (double-array (value-reader idx)) posterior)
          posterior)))))

(def split-rule-lookup-table
  {:gini SplitRule/GINI
   :entropy SplitRule/ENTROPY
   :classification-error  SplitRule/CLASSIFICATION_ERROR})

(def ^:private classifier-metadata
  {:ada-boost
   {:name :ada-boost
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
    :gridsearch-options {:trees (ml-gs/linear 2 50 10 :int64)
                         :max-nodes (ml-gs/linear 4 1000 20 :int64)}
    :property-name-stem "smile.databoost"
    :constructor #(AdaBoost/fit ^Formula %1 ^DataFrame %2 ^Properties %3)
    :predictor tuple-predict-posterior}
   :logistic-regression
   {:name :logistic-regression
    :options [{:name :lambda
               :type :float64
               :default 0.1}
              {:name :tolerance
               :type :float64
               :default 1e-5}
              {:name :max-iter
               :type :int32
               :default 500}]
    :gridsearch-options {:lambda (ml-gs/linear 1e-3 1e2 30)
                         :tolerance (ml-gs/linear 1e-9 1e-1 20)
                         :max-iter (ml-gs/linear 1e2 1e4 20 :int64)}
    :constructor #(LogisticRegression/fit ^Formula %1 ^DataFrame %2 ^Properties %3)
    :predictor double-array-predict-posterior}

   :decision-tree
   {:attributes #{:probabilities :attributes}
    :name :decision-tree
    :options [{:name :max-nodes
               :type :int32
               :default 100}
              {:name :node-size
               :type :int32
               :default 1}
              {:name :max-depth
               :type :int32 
               :default 20}
              {:name :split-rule
               ;; :type :enumeration
               ;; :class-type :string
               :type :string
               :lookup-table split-rule-lookup-table
               :default :gini}]
    :gridsearch-options {:max-nodes (ml-gs/linear 10 1000 30)
                         :node-size (ml-gs/linear 1 20 20)
                         :max-depth (ml-gs/linear 1 50 20 )
                         :split-rule (ml-gs/categorical [:gini :entropy :classification-error] )

                         }
    :property-name-stem "smile.cart"
    :constructor #(DecisionTree/fit ^Formula %1 ^DataFrame %2  ^Properties %3)
    :predictor tuple-predict-posterior

    }

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
   ;;
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


(defmulti ^:private model-type->classification-model
  (fn [model-type] model-type))


(defmethod model-type->classification-model :default
  [model-type]
  (if-let [retval (get classifier-metadata model-type)]
    retval
    (throw (ex-info "Failed to find classification model"
                    {:model-type model-type
                     :available-types (keys classifier-metadata)}))))


(defn- train
  [feature-ds label-ds options]
  (let [entry-metadata (model-type->classification-model
                        (model/options->model-type options))
        target-colname (first (ds/column-names label-ds))
        feature-colnames (ds/column-names feature-ds)
        formula (smile-proto/make-formula (ds-utils/column-safe-name target-colname)
                                          (map ds-utils/column-safe-name
                                               feature-colnames))
        dataset (merge feature-ds
                       (ds/update-columnwise
                        label-ds :all
                        dtype/elemwise-cast :int32))
        data (smile-data/dataset->smile-dataframe dataset)
        properties (smile-proto/options->properties entry-metadata dataset options)
        ctor (:constructor entry-metadata)
        model (ctor formula data properties)]
    (model/model->byte-array model)))


(defn- thaw
  [model-data]
  (model/byte-array->model model-data))


(defn- predict
  [feature-ds thawed-model {:keys [target-columns
                                   target-categorical-maps
                                   options]}]
  (let [entry-metadata (model-type->classification-model
                        (model/options->model-type options))
        target-colname (first target-columns)
        n-labels (-> (get target-categorical-maps target-colname)
                     :lookup-table
                     count)
        predictor (:predictor entry-metadata)
        predictions (predictor thawed-model feature-ds options n-labels)]
    (-> predictions
        (dtt/->tensor)
        (model/finalize-classification (ds/row-count feature-ds)
                                       target-colname
                                       target-categorical-maps))))

(doseq [[reg-kwd reg-def] classifier-metadata]
  (ml/define-model! (keyword "smile.classification" (name reg-kwd))
    train predict {:thaw-fn thaw
                   :hyperparameters (:gridsearch-options reg-def)}))


(comment
  (do
    (require '[tech.v3.dataset.column-filters :as cf])
    (require '[tech.v3.dataset.modelling :as ds-mod])
    (require '[tech.v3.ml.loss :as loss])
    (def src-ds (ds/->dataset "test/data/iris.csv"))
    (def ds (->  src-ds
                 (ds/categorical->number cf/categorical)
                 (ds-mod/set-inference-target "species")))
    (def feature-ds (cf/feature ds))
    (def split-data (ds-mod/train-test-split ds))
    (def train-ds (:train-ds split-data))
    (def test-ds (:test-ds split-data))
    (def model (ml/train train-ds {:model-type :smile.classification/decision-tree
                                   :split-rule SplitRule/CLASSIFICATION_ERROR}))
    (def prediction (ml/predict test-ds model))))
