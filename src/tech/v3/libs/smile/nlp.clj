(ns tech.v3.libs.smile.nlp
  (:require [clojure.string :as str]
            [pppmap.core :as ppp]
            [tech.v3.dataset :as ds])
  (:import smile.nlp.normalizer.SimpleNormalizer
           smile.nlp.stemmer.PorterStemmer
           [smile.nlp.tokenizer SimpleSentenceSplitter SimpleTokenizer]))


(defn default-text->bow [text]
  "Converts text to token counts (a map token -> count)"
  (let [normalizer (SimpleNormalizer/getInstance)
        tokenizer (SimpleTokenizer. )
        sentence-splitter (SimpleSentenceSplitter/getInstance)
        stemmer (PorterStemmer.)]
    (->> text
         (.normalize normalizer)
         (.split sentence-splitter)
         (map #(.split tokenizer %))
         (map seq)
         flatten
         (remove nil?)
         (map #(.stem stemmer %))
         (map str/lower-case)
         frequencies)))


(defn ->vocabulary-top-n [bows n]
  "Takes top-n most frequent tokens"
  (let [vocabulary
        (->>
         (apply merge-with + bows)
         (sort-by second)
         reverse
         (take n)
         keys)
        vocab->index-map (zipmap vocabulary (range))]

    {:vocab vocabulary
     :vocab->index-map vocab->index-map
     :index->vocab-map (clojure.set/map-invert vocab->index-map)
     }))

(defn count-vectorize [ds text-col bow-col text->bow-fn]
  "Converts text column `text-col` to bag-of-words representation
   in the form of a frequency-count map"
  (ds/add-or-update-column
   ds
   (ds/new-column
    bow-col
    (ppp/ppmap-with-progress
     "text->bow"
     1000
     text->bow-fn
     (get ds text-col)))))


(defn bow->something-sparse [ds bow-col indices-col vocab-size bow->sparse-fn]
  "Converts a bag-of-word column `bow-col` to a sparse data column `indices-col`.
   The exact transformation to the sparse representtaion is given by `bow->sparse-fn`"
  (let [vocabulary (->vocabulary-top-n (get ds bow-col) vocab-size)
        vocab->index-map (:vocab->index-map vocabulary)
        ds
        (vary-meta ds assoc
                   :count-vectorize-vocabulary vocabulary)]
    (ds/add-or-update-column
     ds
     (ds/new-column
      indices-col
      (ppp/ppmap-with-progress
       "bow->sparse"
       1000
       #(bow->sparse-fn % vocab->index-map)
       (get ds bow-col))))))
