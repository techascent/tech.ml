(ns tech.v3.libs.smile.nlp
  (:require [clojure.string :as str]
            [pppmap.core :as ppp]
            [tech.v3.dataset :as ds])
  (:import smile.nlp.normalizer.SimpleNormalizer
           smile.nlp.stemmer.PorterStemmer
           [smile.nlp.tokenizer SimpleSentenceSplitter SimpleTokenizer]))

;; )
(defn default-text->bow [text]
  "Converts text to token counts (a map token -> count"
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


(defn ->vocabulary-top-n [ds bow-col n]
  "Takes top-n most frequent tokens"
  (let [vocabulary
        (->>
         (apply merge-with + (get ds bow-col))
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
  "Converts text column `text-col` to bag-of-words crepresentation
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


