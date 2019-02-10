(ns tech.libs.tablesaw-test
  (:require [tech.libs.tablesaw :as tablesaw]
            [tech.libs.tablesaw.datatype.tablesaw :as dtype-tbl]
            [tech.ml.dataset.etl :as etl]
            [tech.ml.dataset.etl.column-filters :as col-filters]
            [tech.ml.dataset :as ds]
            [tech.ml.dataset.column :as ds-col]
            [tech.ml.loss :as loss]
            [tech.ml :as ml]
            [tech.libs.xgboost]
            [tech.datatype :as dtype]
            [clojure.core.matrix :as m]
            [clojure.set :as c-set]
            [clojure.java.io :as io]
            [clojure.string :as s]
            [camel-snake-kebab.core :refer [->kebab-case]]
            [clojure.test :refer :all]))


(deftest tablesaw-col-subset-test
  (let [test-col (tablesaw/make-column
                  (dtype-tbl/make-column :int32 (range 10)) {})
        select-vec [3 5 7 3 2 1]
        new-col (ds-col/select test-col select-vec)]
    (is (= select-vec
           (dtype/->vector new-col)))))


(def basic-pipeline
  '[[remove "Id"]
    ;;Replace missing values or just empty csv values with NA
    [replace-missing string? "NA"]
    [replace-string string? "" "NA"]
    [replace-missing numeric? 0]
    [replace-missing boolean? false]
    [->etl-datatype [or numeric? boolean?]]
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
    [m= "SalePriceDup" (col "SalePrice")]
    [m= "SalePrice" (log1p (col "SalePrice"))]])


(deftest base-etl-test
  (let [src-dataset (tablesaw/path->tablesaw-dataset "data/aimes-house-prices/train.csv")
        ;;For inference, we won't have the target but we will have everything else.
        inference-columns (c-set/difference
                           (set (map ds-col/column-name
                                     (ds/columns src-dataset)))
                           #{"SalePrice"})
        inference-dataset (-> (ds/select src-dataset
                                               inference-columns
                                               (range 10))
                              (ds/->flyweight :error-on-missing-values? false))
        {:keys [dataset pipeline options]}
        (-> src-dataset
            (etl/apply-pipeline basic-pipeline
                                {:target "SalePrice"}))
        post-pipeline-columns (c-set/difference inference-columns #{"Id"})
        sane-dataset-for-flyweight (ds/select dataset post-pipeline-columns
                                                    (range 10))
        final-flyweight (-> sane-dataset-for-flyweight
                            (ds/->flyweight))]
    (is (= [1460 81] (m/shape src-dataset)))
    (is (= [1460 81] (m/shape dataset)))

    (is (= 45
           (count (col-filters/execute-column-filter dataset :categorical?))))
    (is (= #{"MSSubClass" "OverallQual" "OverallCond"}
           (c-set/intersection #{"MSSubClass" "OverallQual" "OverallCond"}
                               (set (col-filters/execute-column-filter dataset :categorical?)))))
    (is (= []
           (vec (col-filters/execute-column-filter dataset :string?))))
    (is (= ["SalePrice"]
           (vec (col-filters/execute-column-filter dataset :target?))))
    (is (= []
           (vec (col-filters/execute-column-filter dataset [:not [:numeric?]]))))
    (let [sale-price (ds/column dataset "SalePriceDup")
          sale-price-l1p (ds/column dataset "SalePrice")
          sp-stats (ds-col/stats sale-price [:mean :min :max])
          sp1p-stats (ds-col/stats sale-price-l1p [:mean :min :max])]
      (is (m/equals (mapv sp-stats [:mean :min :max])
                    [180921.195890 34900 755000]
                    0.01))
      (is (m/equals (mapv sp1p-stats [:mean :min :max])
                    [12.024 10.460 13.534]
                    0.01)))

    (is (= 10 (count inference-dataset)))
    (is (= 10 (count final-flyweight)))

    (let [exact-columns (tablesaw/map-seq->tablesaw-dataset
                         inference-dataset
                         {:column-definitions (:dataset-column-metadata options)})
          ;;Just checking that this works at all..
          autoscan-columns (tablesaw/map-seq->tablesaw-dataset inference-dataset {})]

      ;;And the definition of exact is...
      (is (= (mapv :datatype (->> (:dataset-column-metadata options)
                                  (sort-by :name)))
             (->> (ds/columns exact-columns)
                  (map ds-col/metadata)
                  (sort-by :name)
                  (mapv :datatype))))
      (let [inference-ds (-> (etl/apply-pipeline exact-columns pipeline
                                                 (assoc options :inference? true))
                             :dataset)]
        ;;spot check a few of the items
        (is (m/equals (dtype/->vector (ds/column sane-dataset-for-flyweight "MSSubClass"))
                      (dtype/->vector (ds/column inference-ds "MSSubClass"))))
        ;;did categoical values get encoded identically?
        (is (m/equals (dtype/->vector (ds/column sane-dataset-for-flyweight "OverallQual"))
                      (dtype/->vector (ds/column inference-ds "OverallQual"))))))))


(defn train-test-split
  [dataset & {:keys [train-fraction]
              :or {train-fraction 0.7}}]
  (let [[num-rows num-cols] (m/shape dataset)
        num-rows (long num-rows)
        index-seq (shuffle (range num-rows))
        num-train (long (* num-rows train-fraction))
        train-indexes (take num-train index-seq)
        test-indexes (drop num-train index-seq)]
    {:train-ds (ds/select dataset :all train-indexes)
     :test-ds (ds/select dataset :all test-indexes)}))


(deftest train-predict-test
  (let [src-dataset (tablesaw/path->tablesaw-dataset
                     "data/aimes-house-prices/train.csv")
        {:keys [dataset pipeline options]}
        (etl/apply-pipeline src-dataset basic-pipeline {:target "SalePrice"})
        {:keys [train-ds test-ds]} (train-test-split dataset)
        all-columns (set (map ds-col/column-name (ds/columns dataset)))
        label-keys #{"SalePrice"}
        feature-keys (c-set/difference all-columns #{"SalePrice" "SalePriceDup"})
        test-full-row-major (->> (ds/->row-major dataset {:feature-keys feature-keys
                                                          :label-keys label-keys}
                                                 {:datatype :float32})
                                 (take 10)
                                 vec)

        test-train-row-major (->> (ds/->row-major train-ds {:feature-keys feature-keys
                                                                  :label-keys label-keys}
                                                        {:datatype :float32})
                                  (take 10)
                                  vec)

        test-test-row-major (->> (ds/->row-major train-ds {:feature-keys feature-keys
                                                                 :label-keys label-keys}
                                                       {:datatype :float32})
                                 (take 10)
                                 (vec))
        model (ml/train {:model-type :xgboost/regression}
                        feature-keys label-keys
                        train-ds)
        labels (dtype/->vector (ds/column test-ds "SalePrice"))
        predictions (ml/predict model test-ds)
        loss-value (loss/rmse predictions labels)]
    (is (< loss-value 0.20))))


(def mapseq-fruit-dataset
  (memoize
   (fn []
     (let [fruit-ds (slurp (io/resource "fruit_data_with_colors.txt"))
           dataset (->> (s/split fruit-ds #"\n")
                        (mapv #(s/split % #"\s+")))
           ds-keys (->> (first dataset)
                        (mapv (comp keyword ->kebab-case)))]
       (->> (rest dataset)
            (map (fn [ds-line]
                   (->> ds-line
                        (map (fn [ds-val]
                               (try
                                 (Double/parseDouble ^String ds-val)
                                 (catch Throwable e
                                   (-> (->kebab-case ds-val)
                                       keyword)))))
                        (zipmap ds-keys)))))))))


;;A sequence of maps is actually hard because keywords aren't represented
;;in tablesaw so we have to do a lot of work.  Classification also imposes
;;the necessity of mapping back from the label column to a sequence of
;;keyword labels.
(deftest mapseq-classification-test
  (let [pipeline '[[remove :fruit-subtype]
                   [string->number string?]
                   ;;Range numeric data to -1 1
                   [range-scaler (not categorical?)]]
        src-ds (ds/->dataset (mapseq-fruit-dataset))

        {:keys [dataset pipeline options]}
        (etl/apply-pipeline src-ds pipeline
                            {:target :fruit-name})

        origin-ds (mapseq-fruit-dataset)
        src-keys (set (keys (first (mapseq-fruit-dataset))))
        result-keys (set (->> (ds/columns dataset)
                              (map ds-col/column-name)))]

    ;;Kind of hard
    (is (= (set (keys (first (mapseq-fruit-dataset))))
           (set (->> (ds/columns src-ds)
                     (map ds-col/column-name)))))

    (is (= (c-set/difference src-keys #{:fruit-subtype})
           result-keys))

    ;;Really hard
    (is (= (mapv (comp name :fruit-name) (mapseq-fruit-dataset))
           (ds/labels dataset options)))))
