(ns tech.libs.smile.classification
  (:require [clojure.reflect :refer [reflect]]
            [tech.libs.smile.utils :as utils]
            [tech.libs.smile.kernels]
            [tech.libs.smile.distance]
            [tech.v2.datatype :as dtype]
            [tech.ml.protocols.system :as ml-proto]
            [tech.ml.model :as model]
            [tech.ml.registry :as registry]
            [tech.ml.dataset :as dataset]
            [tech.ml.gridsearch :as ml-gs]
            [tech.ml.dataset.options :as ds-options]
            [camel-snake-kebab.core :refer [->kebab-case]])
  (:import [smile.classification DecisionTree$SplitRule
            NaiveBayes$Model SVM$Multiclass OnlineClassifier SoftClassifier
            Classifier SVM]))



(def package-name "smile.classification")


(def java-classes
  #{
    "AdaBoost"
    "DecisionTree"
    "FLD"
    "GradientTreeBoost"
    "KNN"
    "LDA"
    "LogisticRegression"
    "Maxent"
    "NaiveBayes"
    "NeuralNetwork"
    "PlattScaling"
    "QDA"
    "RandomForest"
    "RBFNetwork"
    "RDA"
    "SVM"
    })


(defn reflect-cls
  [cls-name]
  (reflect (utils/class-name->class package-name
                                    cls-name)))


(def base-attribute-map
  {:probabilities "SoftClassifier"
   :online "OnlineClassifier"})


(defn java-classes->metadata
  []
  (->> java-classes
       (map (fn [cls-name]
              (let [item-name (-> cls-name
                                  ->kebab-case
                                  keyword)
                    reflect-data (reflect-cls cls-name)
                    bases (map str (:bases reflect-data))
                    ]
                [item-name
                 {:class-name cls-name
                  :name item-name
                  :datatypes (utils/method-datatype "predict" reflect-data)
                  :attributes (->> base-attribute-map
                                   (map (fn [[keywd base-cls]]
                                          (when (->> bases
                                                     (filter #(.contains ^String % base-cls))
                                                     seq)
                                            keywd)))
                                   (remove nil?)
                                   set)}])))
       (into {})))



(def classifier-metadata
  {:ada-boost {:attributes #{:probabilities :attributes}
               :class-name "AdaBoost"
               :datatypes #{:float64-array}
               :name :ada-boost
               :options [{:name :ntrees
                          :type :int32
                          :default 200}
                         {:name :max-nodes
                          :type :int32
                          :default 6}]
               :gridsearch-options {:ntrees (ml-gs/linear-long [2 500])
                                    :max-nodes (ml-gs/linear-long [4 1000])}}
   :decision-tree {:attributes #{:probabilities :attributes}
                   :class-name "DecisionTree"
                   :datatypes #{:float64-array}
                   :name :decision-tree
                   :options [{:name :max-nodes
                              :type :int32
                              :default 100}
                             {:name :node-size
                              :type :int32
                              :default 1}
                             {:name :split-rule
                              :type :enumeration
                              :class-type DecisionTree$SplitRule
                              :lookup-table {:gini DecisionTree$SplitRule/GINI
                                             :entropy DecisionTree$SplitRule/ENTROPY
                                             :classification-error DecisionTree$SplitRule/CLASSIFICATION_ERROR}
                              :default :gini}]}
   :fld {:attributes #{:projection}
         :class-name "FLD"
         :datatypes #{:float64-array}
         :name :fld
         :options [{:name :L
                    :type :int32
                    :default -1}
                   {:name :tolerance
                    :type :float64
                    :default 1e-4}]}
   :gradient-tree-boost {:attributes #{:probabilities}
                         :class-name "GradientTreeBoost"
                         :datatypes #{:float64-array}
                         :name :gradient-tree-boost
                         :options [{:name :ntrees
                                    :type :int32
                                    :default 500}
                                   {:name :max-nodes
                                    :type :int32
                                    :default 6
                                    :range :>0}
                                   {:name :shrinkage
                                    :type :float64
                                    :default 0.005
                                    :range :>0}
                                   {:name :sampling-fraction
                                    :type :float64
                                    :default 0.7
                                    :range [0.0 1.0]}]}
   :knn {:attributes #{:probabilities :object-data}
         :class-name "KNN"
         :datatypes #{:float64-array}
         :name :knn
         :options [{:name :distance
                    :type :distance
                    :default {:distance-type :euclidean}}
                   {:name :num-clusters
                    :type :int32
                    :default 5}]
         :gridsearch-options {:num-clusters (ml-gs/linear-long [2 100])}}
   :logistic-regression {:attributes #{:probabilities}
                         :class-name "LogisticRegression"
                         :datatypes #{:float64-array}
                         :name :logistic-regression
                         :options [{:name :lambda
                                    :type :float64
                                    :default 0.1}
                                   {:name :tolerance
                                    :type :float64
                                    :default 1e-5}
                                   {:name :max-iter
                                    :type :int32
                                    :default 500}]
                         :gridsearch-options {:lambda (ml-gs/exp [1e-3 1e2])
                                              :tolerance (ml-gs/linear [1e-9 1e-1])
                                              :max-iter (ml-gs/linear-long [1e2 1e4])}}
   ;;Not supported at this time because constructor patter is unique
   :maxent {:attributes #{:probabilities}
            :class-name "Maxent"
            :datatypes #{:float64-array :int32-array}
            :name :maxent}

   :naive-bayes {:attributes #{:online :probabilities}
                 :class-name "NaiveBayes"
                 :datatypes #{:float64-array :sparse}
                 :name :naive-bayes
                 :options [{:name :model
                            :type :enumeration
                            :class-type NaiveBayes$Model
                            :lookup-table {
                                           ;; Users have to provide probabilities for this to work.
                                           ;; :general NaiveBayes$Model/GENERAL

                                           :multinomial NaiveBayes$Model/MULTINOMIAL
                                           :bernoulli NaiveBayes$Model/BERNOULLI
                                           :polyaurn NaiveBayes$Model/POLYAURN}
                            :default :multinomial}
                           {:name :num-classes
                            :type :int32
                            :default utils/options->num-classes}
                           {:name :input-dimensionality
                            :type :int32
                            :default utils/options->feature-ecount}
                           {:name :sigma
                            :type :float64
                            :default 1.0}]
                 :gridsearch-options {:model (ml-gs/nominative [:multinomial :bernoulli :polyaurn])
                                      :sigma (ml-gs/exp [1e-4 0.2])}}
   :neural-network {:attributes #{:online :probabilities}
                    :class-name "NeuralNetwork"
                    :datatypes #{:float64-array}
                    :name :neural-network}
   :platt-scaling {:attributes #{}
                   :class-name "PlattScaling"
                   :datatypes #{:double}
                   :name :platt-scaling}


   ;;Lots of discriminant analysis
   :linear-discriminant-analysis
   {:attributes #{:probabilities}
    :class-name "LDA"
    :datatypes #{:float64-array}
    :name :lda
    :options [{:name :prioiri
               :type :float64-array
               :default nil}
              {:name :tolerance
               :default 1e-4
               :type :float64}]
    :gridsearch-options {:tolerance (ml-gs/linear [1e-9 1e-2])}}


   :quadratic-discriminant-analysis
   {:attributes #{:probabilities}
    :class-name "QDA"
    :datatypes #{:float64-array}
    :name :qda
    :options [{:name :prioiri
               :type :float64-array
               :default nil}
              {:name :tolerance
               :default 1e-4
               :type :float64}]
    :gridsearch-options {:tolerance (ml-gs/linear [1e-9 1e-2])}}


   :regularized-discriminant-analysis
   {:attributes #{:probabilities}
    :class-name "RDA"
    :datatypes #{:float64-array}
    :name :rda
    :options [{:name :prioiri
               :type :float64-array
               :default nil}
              {:name :alpha
               :type :float64
               :default 0.0 }
              {:name :tolerance
               :default 1e-4
               :type :float64}]
    :gridsearch-options {:tolerance (ml-gs/linear [1e-9 1e-2])
                         :alpha (ml-gs/linear [0.0 1.0])}}


   :random-forest {:attributes #{:probabilities}
                   :class-name "RandomForest"
                   :datatypes #{:float64-array}
                   :name :random-forest}
   :rbf-network {:attributes #{}
                 :class-name "RBFNetwork"
                 :datatypes #{}
                 :name :rbf-network}


   :svm {:attributes #{:online :probabilities}
         :class-name "SVM"
         :datatypes #{:float64-array}
         :name :svm
         :constructor-filter (fn [options mixed-data-entry]
                               ;;There is a different constructor when the number of classes is 2
                               (if (> (utils/options->num-classes options)
                                      2)
                                 mixed-data-entry
                                 (let [opt-name (-> (nth mixed-data-entry 2)
                                                    :name)]
                                   (when-not (or (= opt-name :multiclass-strategy)
                                                 (= opt-name :num-classes))
                                       mixed-data-entry))))
         :options [{:name :kernel
                    :type :mercer-kernel
                    :default {:kernel-type :gaussian}}
                   {:name :soft-margin-penalty
                    :type :float64
                    :altname "C"
                    :default 1.0}
                   {:name :num-classes
                    :type :int32
                    :default utils/options->num-classes}
                   {:name :multiclass-strategy
                    :type :enumeration
                    :class-type SVM$Multiclass
                    :lookup-table {:one-vs-one SVM$Multiclass/ONE_VS_ONE
                                   :one-vs-all SVM$Multiclass/ONE_VS_ALL}
                    :default :one-vs-one}]
         :gridsearch-options {:kernel {:kernel-type (ml-gs/nominative [:gaussian :linear])}
                              :soft-margin-penalty (ml-gs/exp [1e-4 1e2])
                              :multiclass-strategy (ml-gs/nominative [:one-vs-one
                                                                      :one-vs-all])}}})


(defn model-type->classification-model
  [model-type]
  (if-let [retval (get classifier-metadata model-type)]
    retval
    (throw (ex-info "Unrecognized model type"
                    {:model-type model-type
                     :available-types (keys classifier-metadata)}))))



(defn- train-online
  "Online systems can train iteratively.  They can handle therefore much larger
  datasets."
  [options entry-metadata row-major-dataset]
  (let [;;Do basic NN shit to make it work.  Users don't need to specify the
        ;;parts that are dataset specific (input-size) *or* that never change
        ;;(output-size).
        ^OnlineClassifier untrained
        (-> (utils/prepend-data-constructor-arguments entry-metadata options [])
            (utils/construct package-name options))]
    (->> row-major-dataset
         (map #(.learn untrained ^doubles (:features %)
                       (int (dtype/get-value (:label %) 0))))
         dorun)
    (when (= (:name entry-metadata) :svm)
      (let [^SVM sort-of-trained untrained]
        (.trainPlattScaling sort-of-trained
                            (->> (map :features row-major-dataset)
                                 object-array)
                            ^ints
                            (->> (map (comp int #(dtype/get-value % 0) :label) row-major-dataset)
                                 int-array))))
    ;;its trained now
    untrained))


(defn- train-block
  "Train by downloading all the data into a fixed matrix."
  [options entry-metadata row-major-dataset]
  (let [value-seq (->> row-major-dataset
                       (map :features))
        [x-data x-datatype] (if (contains? (:attributes entry-metadata)
                                           :object-data)
                              [(object-array value-seq) :object-array]
                              [(into-array value-seq) :float64-array-array])

        n-entries (first (dtype/shape x-data))
        ^ints y-data (first (dtype/copy-raw->item!
                             (map :label row-major-dataset)
                             (dtype/make-array-of-type :int32 n-entries)
                             0))
        data-constructor-arguments [{:type x-datatype
                                     :default x-data
                                     :name :training-data}
                                    {:type :int32-array
                                     :default y-data
                                     :name :labels}]]
    (-> (utils/prepend-data-constructor-arguments entry-metadata options
                                                  data-constructor-arguments)
        (utils/construct package-name options))))


(defrecord SmileClassification []
  ml-proto/PMLSystem
  (system-name [_] :smile.classification)
  (gridsearch-options [system options]
    (let [entry-metadata (model-type->classification-model (model/options->model-type options))]
      (if-let [retval (:gridsearch-options entry-metadata)]
        retval
        (throw (ex-info "Model type does not support auto gridsearch yet"
                        {:entry-metadata entry-metadata})))))
  (train [system options dataset]
    (let [entry-metadata (model-type->classification-model (model/options->model-type options))
          row-major-dataset (dataset/->row-major dataset options)]
      (-> (if (contains? (:attributes entry-metadata) :online)
            (train-online options entry-metadata row-major-dataset)
            (train-block options entry-metadata row-major-dataset))
          model/model->byte-array)))
  (predict [system options trained-model-bytes dataset]
    (let [row-major-dataset (dataset/->row-major dataset options)
          trained-model (model/byte-array->model trained-model-bytes)
          inverse-label-map (ds-options/inference-target-label-inverse-map options)
          ordered-labels (->> inverse-label-map
                              (sort-by first)
                              (mapv second))]
      (if (instance? SoftClassifier trained-model)
        (let [probabilities (double-array (count ordered-labels))
              ^SoftClassifier trained-model trained-model]
          (->> row-major-dataset
               (map (fn [{:keys [:features]}]
                      (.predict trained-model ^doubles features probabilities)
                      (zipmap ordered-labels probabilities)))))
        (let [^Classifier trained-model trained-model]
          (->> row-major-dataset
               (map (fn [{:keys [:features]}]
                      (let [prediction (.predict trained-model ^doubles features)]
                        {(get ordered-labels prediction) 1.0})))))))))



(def system (constantly (->SmileClassification)))


(registry/register-system (system))
