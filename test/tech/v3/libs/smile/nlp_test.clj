(ns tech.v3.libs.smile.nlp-test
  (:require [tech.v3.libs.smile.nlp :as nlp]
            [tech.v3.dataset :as ds]
            [clojure.test :refer [deftest is] :as t]))

(deftest calculate-tfidf

  (let [bows [{:this 1 :is 1 :a 2 :sample 1}
              {:this 1 :is 1 :another 2 :example 3 }]
        tf-map (nlp/tf-map bows)
        bow-1 (first bows)
        bow-2 (second bows)]
    (is (= 0.0  (nlp/tfidf tf-map :example bow-1  bows)))
    (is (= 0.12901285528456338  (nlp/tfidf tf-map :example bow-2  bows)))))


(deftest bow->tfidf
  (let [ds (->
            (ds/->dataset {:text ["This is a a sample"  "this is another another example example example" ]})
            (nlp/count-vectorize :text :bow nlp/default-text->bow)
            (nlp/bow->tfidf :bow :tfidf)
            )
        tfidf-1 (first (:tfidf ds))
        tfidf-2 (second (:tfidf ds))
        ]
    (is (= 0.12901285528456338 (get tfidf-2 "exampl")))
    (is (= 0.12041199826559248 (get tfidf-1 "a")))
    (is (= 0.0 (get tfidf-1 "thi")))))
